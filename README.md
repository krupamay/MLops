Install Conda  
create -n "my_new_env" python=3.10  
conda activate my_new_env  
pip install -r requirements.txt    
python plot_digits_classification.py  

# Named Argument from Command line

python plot_digits_classification.py  --test_sizes 0.1,0.2 --dev_sizes 0.1,0.2 --gamma_list 0.001,0.01,0.1 --C_list 1,10,100



