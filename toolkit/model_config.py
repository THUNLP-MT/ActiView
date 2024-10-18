import transformers

#################################
"""
Configuration of  model path
"""
model_mapper = {'idefics': 'idefics2-8b',
            'idefics3': 'Idefics3-8B-Llama3',
            'llavanext': 'lmms-lab/llava-next-interleave-qwen-7b',
            'llavaonevision': 'llava-onevision-qwen2-7b-ov',
            'mantis': 'TIGER-Lab/Mantis-8B-Idefics2',
            'minicpmv2.5':'MiniCPM-Llama3-V-2_5',
            'minicpmv2.6':'MiniCPM-V-2_6',
            'mplugowl3': 'mplug_owl3',
            'mplugowl2': 'mplug-owl2-llama2-7b',
            'qwen2vl': 'Qwen2-VL-7B-Instruct',
            'slime8b': ''}

#################################
"""
load your model here
"""
def load_model(model_path, model_scale, processpor_path, model_name='Brote'):
    print('##########')
    print(model_name)
    print('##########')
    if model_name.lower() in model_mapper:
        tokenizer, processor, generation_kwargs = None, None, None
        model_name = model_name.lower()
        model_path = model_mapper[model_name]

        #global tokenizer
        if model_name == 'mplugowl2':
            from model_mplugowl2 import load_model as load_model_mplugowl2
            from model_mplugowl2 import inference as inference_mplugowl2

            model_generation = load_model_mplugowl2('mplug-owl2-llama2-7b')

        elif model_name == 'mplugowl3':
            sys.path.append("/yeesuanAI05/thumt/dyr/mplug_owl3")
            from mplug_owl3.modeling_mplugowl3 import mPLUGOwl3Model
            from transformers import AutoTokenizer

            model = mPLUGOwl3Model.from_pretrained(model_path, attn_implementation='sdpa', torch_dtype=torch.half).cuda()
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            processor = model.init_processor(tokenizer)
            generation_kwargs = {'tokenizer': tokenizer}

        elif model_name == 'minicpmv2.6':
            from pathlib import Path
            sys.path.append(str(mini_cpm_path))
            from transformers import AutoModel, AutoTokenizer

            torch.manual_seed(0)
            model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to('cuda')
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            generation_kwargs = {'tokenizer': tokenizer}

        elif model_name == 'minicpmv2.5':
            sys.path.append(str(mini_cpm_path))
            from chat import MiniCPMVChat

            torch.manual_seed(0)
            model = MiniCPMVChat(model_path)

        elif model_name == 'qwen2vl':
            from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

            model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
            processor = AutoProcessor.from_pretrained(model_path, max_pixels = 1040 * 28 * 28, min_pixels = 100 * 28 * 28)
            
        elif 'llava' in model_name: # ov, TODO: other llava models
            from llava.model.builder import load_pretrained_model
            if 'onevision' in model_name:
                warnings.filterwarnings("ignore")
                from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
                from llava.conversation import conv_templates
                #tokenizer, model, image_processor, max_length = load_pretrained_model(model_path, None, 'llava_qwen', device_map='auto', force_download=False)  # Add any other thing you want to pass in llava_model_args
                tokenizer, model, image_processor, max_length = load_pretrained_model(model_path, None, 'llava_qwen', device_map='auto', force_download=False, image_aspect_ratio="anyres")
                #tokenizer, model, image_processor, max_length = load_pretrained_model(model_path, None, 'llava_qwen', device_map='auto', force_download=False, image_aspect_ratio="square")
                generation_kwargs = {
                    'IMAGE_TOKEN_INDEX': IMAGE_TOKEN_INDEX,
                    'DEFAULT_IMAGE_TOKEN': DEFAULT_IMAGE_TOKEN,
                    'conv_templates': conv_templates,
                    'conv_template': "qwen_2",  # Make sure you use correct chat template for different models
                    'tokenizer': tokenizer,
                    'image_processor': image_processor}
        else:
            # mantis, idefics, idefics3
            from transformers import AutoProcessor, AutoModelForVision2Seq

            #processor = AutoProcessor.from_pretrained("TIGER-Lab/Mantis-8B-Idefics2") # do_image_splitting is False by default
            #model = AutoModelForVision2Seq.from_pretrained("TIGER-Lab/Mantis-8B-Idefics2", device_map="auto").to(torch.bfloat16)
            if model_name.startswith('idefics'):
                processor = AutoProcessor.from_pretrained(model_path, do_image_splitting=False) 
            else:
                print('loading')
                processor = AutoProcessor.from_pretrained(model_path) # do_image_splitting is False by default
                # model = AutoModelForVision2Seq.from_pretrained(model_mapper[model_name], device_map="auto").to(torch.bfloat16)
            model = AutoModelForVision2Seq.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)

            generation_kwargs = {
                "max_new_tokens": 20,
                "num_beams": 1,
                "do_sample": False}
        return model.eval(), processor, generation_kwargs

    if model_scale == 'xxl':
        if model_name == 'MMICL':
            model_ckpt = ""    
        else:
            model_ckpt = ""    
        processor_ckpt = ""
    else:
        if model_name == 'MMICL':
            model_ckpt = brote_root+''
        else:
            model_ckpt = brote_root+''
        processor_ckpt = brote_root+''
    
    sys.path.append(brote_root+'/MMICL/MIC-master')
    if not 'MMICL' in  model_ckpt:
        from model.instructblip_icl import InstructBlipConfig, InstructBlipForConditionalGeneration, InstructBlipProcessor
    else:
        from model.instructblip import InstructBlipConfig, InstructBlipModel, InstructBlipPreTrainedModel,InstructBlipForConditionalGeneration,InstructBlipProcessor
    
    config = InstructBlipConfig.from_pretrained(model_ckpt)
    if not 'MMICL' in  model_ckpt:
        config.qformer_config.global_calculation = 'add'
        config.qformer_config.send_condition_to_llm = False
        config.qformer_config.both_condition = False
    
    print("loading models")
    processor = InstructBlipProcessor.from_pretrained(processor_ckpt)
    model = InstructBlipForConditionalGeneration.from_pretrained(model_ckpt,
            config=config).to('cuda:0',dtype=torch.bfloat16) 
    
    image_placeholder="图"
    sp = [image_placeholder]+[f"<image{i}>" for i in range(20)]
    sp = sp+processor.tokenizer.additional_special_tokens[len(sp):]
    processor.tokenizer.add_special_tokens({'additional_special_tokens':sp})
    if model.qformer.embeddings.word_embeddings.weight.shape[0] != len(processor.qformer_tokenizer):
        model.qformer.resize_token_embeddings(len(processor.qformer_tokenizer))
    
    replace_token="".join(32*[image_placeholder])
    return model, processor, replace_token

################################
"""
implement the inference method and post processing methods of your model
"""

def predict(model, processor, images, prompt, model_name="GPT", mode=None):
    if model_name=="minicpmv2.5":
        preds = []
        for idx, (image, prompt) in enumerate(zip(images, prompt)):
        
            msgs = [{"role": "user", "content": prompt}]
            inputs = {"image": pil_image_to_base64(image), "question": json.dumps(msgs)}
        
            generated_text = model.chat(inputs, sampling=False)
            preds.append(generated_text)
            
            if mode == "json":
                try:
                    print(f"Select-{idx}: " + json.dumps(json.loads(generated_text)), flush=True)
                except:
                    print(generated_text)
            else:
                print(f"Answer-{idx}: " + generated_text, flush=True)
        return preds
    else:
        inputs = processor(images=images, text=prompt, return_tensors="pt", padding=True)
        
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
        inputs['img_mask'] = torch.tensor([[1 for i in range(len(images))]])
        inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
        
        #print("predicting")
        inputs = inputs.to('cuda:0')
        
        outputs = model.generate(
                pixel_values = inputs['pixel_values'],
                input_ids = inputs['input_ids'],
                attention_mask = inputs['attention_mask'],
                img_mask = inputs['img_mask'],
                do_sample=False,
                max_length=50,
                min_length=1,
                set_min_padding_size =False,
        )
        
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)
        #print(generated_text)
        return generated_text
