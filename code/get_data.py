import requests
import getpass
def get_data_from_kaggle(data_urls,local_filenames):   
    user_name = input("user_name:")
    if not user_name:
        user_name = 'xiaofeiwen90@gmail.com'  
    password = getpass.getpass('password:')
    kaggle_info = {'UserName':user_name, 'Password':password}
    for idx in range(len(data_urls)):
        data_url = data_urls[idx]
        local_filename = local_filenames[idx]
        r = requests.get(data_url)
        r = requests.post(r.url, data = kaggle_info)
        with open(local_filename, 'bw') as f:
            for chunk in r.iter_content(chunk_size = 512 * 1024):
                if chunk:
                    f.write(chunk)
                    
train_url = 'http://www.kaggle.com/c/digit-recognizer/download/train.csv'
test_url = 'http://www.kaggle.com/c/digit-recognizer/download/test.csv'
get_data_from_kaggle([train_url,test_url],['../input/train.csv','../input/test.csv'])
