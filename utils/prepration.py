from trainer.BaseTransformerTrainer import BASETranformer_Trainer, BASE_Transformer

def build_trainer(config, writer):
    if config.model["name"] == "resnet18":
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                               with_pos='learned', enc_depth=1, dec_depth=8)

        trainer = BASETranformer_Trainer(model=net, config=config, writer=writer)
        return trainer