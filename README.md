Install Conda  
create -n "my_new_env" python=3.10  
conda activate my_new_env  
pip install -r requirements.txt    
python plot_digits_classification.py  

# Named Argument from Command line

python plot_digits_classification.py  --test_sizes 0.1,0.2 --dev_sizes 0.1,0.2 --gamma_list 0.001,0.01,0.1 --C_list 1,10,100

# Docker Commands
Build Docker Image : docker build -t digits:v1 -f docker/Dockerfile .  
List Docker Images : docker images  
Execute Docker image and enter shell : docker run -it digits:v1 bash  
Execute Docker Image : docker run digits:v1  
List Down Docker containers : docker ps



# Flask App Execution Steps
export FLASK_APP=app  
export LC_ALL=en_US.UTF-8  
export LANG=en_US.UTF-8  
flak run  

# Sample CURL POST command 
curl -X POST -H "Content-Type: application/json" \
-d '{"x": "5", "y": "30"}' http://127.0.0.1:5000/model/predict  



