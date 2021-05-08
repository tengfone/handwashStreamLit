mkdir -p ~/.streamlit/
wget https://storage.googleapis.com/dl-big-project/alexnet_128.pt && mv alexnet_128.pt /app/handwashWHO/machine_learning/model/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
