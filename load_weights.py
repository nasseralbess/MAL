from huggingface_hub import snapshot_download
from dotenv import load_dotenv
import os
load_dotenv()
hf_token = os.getenv("hf_token")
# snapshot_download(
#         repo_id="meta-llama/Llama-2-7b-chat-hf",
#         local_dir=f"models/Llama-2-7b-chat-hf",
#         local_dir_use_symlinks=False,
#         token = hf_token,
#         cache_dir = "cache"
# )

# snapshot_download(
#         repo_id="FinGPT/fingpt-forecaster_dow30_llama2-7b_lora",
#         local_dir=f"models/fingpt-forecaster_dow30_llama2-7b_lora",
#         local_dir_use_symlinks=False,
#         token = hf_token,
#         cache_dir = "cache"
# )
# snapshot_download(
#         repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
#         local_dir=f"models/Meta-Llama-3-8B-Instruct",
#         local_dir_use_symlinks=False,
#         token = hf_token,
#         cache_dir = "cache"



        
# )