FROM python:3.8 as base

WORKDIR /app

COPY requirements.txt .
RUN pip install numpy
RUN pip install -r requirements.txt

# =======================================

FROM base as setup

COPY libs .
COPY tools .
RUN cd tools && (echo "all" | python setup.py)
COPY . .

# =======================================

FROM setup as pretrain

RUN wget -nv https://i.imgur.com/bOZ5g1D.jpg -O test.jpg
RUN python main.py -i test.jpg -o test-out.png

# =======================================

FROM pretrain as final

ENTRYPOINT [ "python", "main.py" ]
