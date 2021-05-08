mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
cd ~/machine_learning/model
wget https://storage.googleapis.com/dl-big-project/alexnet_128.pt
