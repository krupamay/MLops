FROM python:3.9.17
WORKDIR /digits
COPY . /digits/
RUN pip3 install --no-cache-dir -r /digits/requirements.txt
#VOLUME ["/digits"]

EXPOSE 8000
#ENV FLASK_APP=api/app
#CMD ["flask", "run", "--host=0.0.0.0"]
CMD ["python", "api/app.py"]

#CMD ["pytest"]
#CMD ["python", "plot_digits_classification.py", "--test_sizes", "0.1,0.2", "--dev_sizes", "0.1,0.2", "--gamma_list", "0.001,0.01,0.1", "--C_list", "1,10,100"]