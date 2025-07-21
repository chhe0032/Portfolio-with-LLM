# Increase timeout (default is 30s)
timeout = 120  

# Restart workers after N requests to prevent memory leaks
max_requests = 100  
max_requests_jitter = 20

# Add this for tokenizers warning
raw_env = ["TOKENIZERS_PARALLELISM=false"]