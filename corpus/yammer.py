import yampy
import json

# authenticator = yampy.Authenticator(client_id='ePFomlHGwySNVOXzrc3A',
#                                     client_secret='j4MCTTvGRQkxQFzkjGvudLp81xEITkIjvv4pub2QY0')
# redirect_uri = "http://localhost:8081"

# auth_url = authenticator.authorization_url(redirect_uri=redirect_uri)
# access_token = authenticator.fetch_access_token(authenticator)
# access_data = authenticator.fetch_access_data('{access token of yammer}')
'''
使用yammer的SDK yampy抓取原始语料保存到文件
使用python2
yampy在python3下运行有问题
'''
yammer = yampy.Yammer(access_token='{access token of yammer}')
# limit调节一次获取的消息条数
messages = yammer.messages.all(limit=20)
json = json.dumps(messages)
with open('./raw.json', 'w') as f:
    f.write(json)
    f.close()