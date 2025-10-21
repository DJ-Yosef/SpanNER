from modelscope.hub.file_download import model_file_download

file_lis = []
with open('files.txt', 'r', encoding='utf-8') as f:
    for line in f:
        url = line.strip()
        file_name = url.split('/')[-1]
        file_lis.append(file_name)
print(file_lis)


model_dir = model_file_download(model_id='Xunzillm4cc/Xunzi-Qwen1.5-7B_chat',
                                file_path='qwq-32b-q4_k_m.gguf',
                                cache_dir='.qwen',
                                local_dir='.qwen/Xunzillm4cc/Xunzi-Qwen1.5-7B_chat',
                                revision='master')