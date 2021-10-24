

data_stream_file_path = './data_stream_file'

def start_check(data_stream_status_idx):
    status_file = open(data_stream_file_path, 'r')
    data_stream_status = status_file.readline().strip()
    status_file.close()
    if data_stream_status == data_stream_status_idx:
        return True
    else:
        print('Current data stream status is wrong:', data_stream_status)
        return False

def end_check(data_stream_status, note_log, start_time, end_time):
    status_file = open(data_stream_file_path, 'w')
    status_file.write(data_stream_status)
    status_file.write('\n')

    status_file.write('Status: ' + note_log)
    status_file.write('\n')

    status_file.write('Start time: ' + start_time)
    status_file.write('\n')

    status_file.write('End time: ' + end_time)
    status_file.write('\n')
    status_file.close()
