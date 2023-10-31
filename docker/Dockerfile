FROM python:3.9.17
COPY . /digits/
RUN pip3 install --no-cache-dir -r /digits/requirements.txt
WORKDIR /digits
CMD ["pytest"]
CMD ["python", "plot_digits_classification.py", "--test_sizes", "0.1,0.2", "--dev_sizes", "0.1,0.2", "--gamma_list", "0.001,0.01,0.1", "--C_list", "1,10,100"]