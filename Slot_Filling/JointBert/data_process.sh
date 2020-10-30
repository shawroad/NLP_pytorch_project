python -m data_process \
  --mode 'train' \
  $@


python -m data_process \
  --mode 'dev' \
  $@


python -m data_process \
  --mode 'test' \
  $@
