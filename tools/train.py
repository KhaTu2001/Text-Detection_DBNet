from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

import yaml
import paddle
import paddle.distributed as dist

from ppocr.data import build_dataloader, set_signal_handlers
from ppocr.modeling.architectures import build_model
from ppocr.losses import build_loss
from ppocr.optimizer import build_optimizer
from ppocr.postprocess import build_post_process
from ppocr.metrics import build_metric
from ppocr.utils.save_load import load_model
from ppocr.utils.utility import set_seed
from ppocr.modeling.architectures import apply_to_static
import tools.program as program

dist.get_world_size()


def main(config, device, logger, vdl_writer, seed):
    
    if config["Global"]["distributed"]:
        dist.init_parallel_env()

    global_config = config["Global"]

    set_signal_handlers()
    train_dataloader = build_dataloader(config, "Train", device, logger, seed)
    if len(train_dataloader) == 0:
        logger.error(
            "Không có hình ảnh trong tập dữ liệu huấn luyện, hãy đảm bảo\n"
            + "\t1. Số lượng hình ảnh trong file danh sách nhãn huấn luyện phải lớn hơn hoặc bằng kích thước batch.\n"
            + "\t2. File chú thích và đường dẫn trong file cấu hình được cung cấp chính xác."
        )
        return

    if config["Eval"]:
        valid_dataloader = build_dataloader(config, "Eval", device, logger, seed)
    else:
        valid_dataloader = None
    step_pre_epoch = len(train_dataloader)

    post_process_class = build_post_process(config["PostProcess"], global_config)

    if hasattr(post_process_class, "character"):
        char_num = len(getattr(post_process_class, "character"))
        if config["Architecture"]["algorithm"] in [
            "Distillation",
        ]: 
            for key in config["Architecture"]["Models"]:
                if (
                    config["Architecture"]["Models"][key]["Head"]["name"] == "MultiHead"
                ):  
                    if config["PostProcess"]["name"] == "DistillationSARLabelDecode":
                        char_num = char_num - 2
                    if config["PostProcess"]["name"] == "DistillationNRTRLabelDecode":
                        char_num = char_num - 3
                    out_channels_list = {}
                    out_channels_list["CTCLabelDecode"] = char_num
    
                    if (
                        list(config["Loss"]["loss_config_list"][-1].keys())[0]
                        == "DistillationSARLoss"
                    ):
                        config["Loss"]["loss_config_list"][-1]["DistillationSARLoss"][
                            "ignore_index"
                        ] = (char_num + 1)
                        out_channels_list["SARLabelDecode"] = char_num + 2
                    elif any(
                        "DistillationNRTRLoss" in d
                        for d in config["Loss"]["loss_config_list"]
                    ):
                        out_channels_list["NRTRLabelDecode"] = char_num + 3

                    config["Architecture"]["Models"][key]["Head"][
                        "out_channels_list"
                    ] = out_channels_list
                else:
                    config["Architecture"]["Models"][key]["Head"][
                        "out_channels"
                    ] = char_num
        elif config["Architecture"]["Head"]["name"] == "MultiHead":  
            if config["PostProcess"]["name"] == "SARLabelDecode":
                char_num = char_num - 2
            if config["PostProcess"]["name"] == "NRTRLabelDecode":
                char_num = char_num - 3
            out_channels_list = {}
            out_channels_list["CTCLabelDecode"] = char_num
  
            if list(config["Loss"]["loss_config_list"][1].keys())[0] == "SARLoss":
                if config["Loss"]["loss_config_list"][1]["SARLoss"] is None:
                    config["Loss"]["loss_config_list"][1]["SARLoss"] = {
                        "ignore_index": char_num + 1
                    }
                else:
                    config["Loss"]["loss_config_list"][1]["SARLoss"]["ignore_index"] = (
                        char_num + 1
                    )
                out_channels_list["SARLabelDecode"] = char_num + 2
            elif list(config["Loss"]["loss_config_list"][1].keys())[0] == "NRTRLoss":
                out_channels_list["NRTRLabelDecode"] = char_num + 3
            config["Architecture"]["Head"]["out_channels_list"] = out_channels_list
        else: 
            config["Architecture"]["Head"]["out_channels"] = char_num

        if config["PostProcess"]["name"] == "SARLabelDecode": 
            config["Loss"]["ignore_index"] = char_num - 1

    model = build_model(config["Architecture"])

    use_sync_bn = config["Global"].get("use_sync_bn", False)
    if use_sync_bn:
        model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logger.info("convert_sync_batchnorm")

    model = apply_to_static(model, config, logger)

    loss_class = build_loss(config["Loss"])

    optimizer, lr_scheduler = build_optimizer(
        config["Optimizer"],
        epochs=config["Global"]["epoch_num"],
        step_each_epoch=len(train_dataloader),
        model=model,
    )

    eval_class = build_metric(config["Metric"])

    logger.info("train dataloader has {} iters".format(len(train_dataloader)))
    if valid_dataloader is not None:
        logger.info("valid dataloader has {} iters".format(len(valid_dataloader)))

    use_amp = config["Global"].get("use_amp", False)
    amp_level = config["Global"].get("amp_level", "O2")
    amp_dtype = config["Global"].get("amp_dtype", "float16")
    amp_custom_black_list = config["Global"].get("amp_custom_black_list", [])
    amp_custom_white_list = config["Global"].get("amp_custom_white_list", [])
    if use_amp:
        AMP_RELATED_FLAGS_SETTING = {
            "FLAGS_max_inplace_grad_add": 8,
        }
        if paddle.is_compiled_with_cuda():
            AMP_RELATED_FLAGS_SETTING.update(
                {
                    "FLAGS_cudnn_batchnorm_spatial_persistent": 1,
                    "FLAGS_gemm_use_half_precision_compute_type": 0,
                }
            )
        paddle.set_flags(AMP_RELATED_FLAGS_SETTING)
        scale_loss = config["Global"].get("scale_loss", 1.0)
        use_dynamic_loss_scaling = config["Global"].get(
            "use_dynamic_loss_scaling", False
        )
        scaler = paddle.amp.GradScaler(
            init_loss_scaling=scale_loss,
            use_dynamic_loss_scaling=use_dynamic_loss_scaling,
        )
        if amp_level == "O2":
            model, optimizer = paddle.amp.decorate(
                models=model,
                optimizers=optimizer,
                level=amp_level,
                master_weight=True,
                dtype=amp_dtype,
            )
    else:
        scaler = None

    pre_best_model_dict = load_model(
        config, model, optimizer, config["Architecture"]["model_type"]
    )

    if config["Global"]["distributed"]:
        model = paddle.DataParallel(model)

    program.train(
        config,
        train_dataloader,
        valid_dataloader,
        device,
        model,
        loss_class,
        optimizer,
        lr_scheduler,
        post_process_class,
        eval_class,
        pre_best_model_dict,
        logger,
        step_pre_epoch,
        vdl_writer,
        scaler,
        amp_level,
        amp_custom_black_list,
        amp_custom_white_list,
        amp_dtype,
    )


def test_reader(config, device, logger):
    loader = build_dataloader(config, "Train", device, logger)
    import time

    starttime = time.time()
    count = 0
    try:
        for data in loader():
            count += 1
            if count % 1 == 0:
                batch_time = time.time() - starttime
                starttime = time.time()
                logger.info(
                    "reader: {}, {}, {}".format(count, len(data[0]), batch_time)
                )
    except Exception as e:
        logger.info(e)
    logger.info("finish reader: {}, Success!".format(count))


if __name__ == "__main__":
    config, device, logger, vdl_writer = program.preprocess(is_train=True)
    seed = config["Global"]["seed"] if "seed" in config["Global"] else 1024
    set_seed(seed)
    main(config, device, logger, vdl_writer, seed)

