Install all the dependencies
!pip install -q -U bitsandbytes
!pip install -q -U git+https://github.com/huggingface/transformers.git
!pip install -q -U git+https://github.com/huggingface/peft.git
!pip install -q -U git+https://github.com/huggingface/accelerate.git
!pip install -q -U datasets scipy ipywidgets matplotlib

To get the sales KRA data run this command:
!git clone https://huggingface.co/datasets/AdiOO7/SalesKRA
