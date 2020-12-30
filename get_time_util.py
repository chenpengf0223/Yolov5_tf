import datetime

def get_last_time():
  return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

