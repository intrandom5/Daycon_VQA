from transformers import T5Config, AutoModelForCausalLM
from modeling_t5 import VLT5

    
def get_vlt5(model_name):
    t5_config = T5Config.from_pretrained("google/flan-t5-base")
    t5_config.feat_dim = 2048
    t5_config.pos_dim = 4
    t5_config.n_images = 36
    t5_config.individual_vis_layer_norm = True
    t5_config.use_vis_layer_norm = True
    t5_config.use_vis_order_embedding = True
    t5_config.tie_word_embeddings=True
    return VLT5.from_pretrained(model_name, config=t5_config)

def get_git():
    return AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")
