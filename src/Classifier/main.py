import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'tsingenda.settings')
import django
django.setup()
import time
from tsingenda_app.models import ConfData, AgendaData, User
from src.Classifier import train as src_train
from src.Classifier import confidence as src_confidence
from django.conf import settings
from src.Classifier.utils import load_file_to_db


update_int = 60
update_data_inc = 1
show_int = 30
agenda_model_base = '/root/work/Tsingenda-backend/src/Classifier/checkpoint'
conf_model_base = '/root/work/Tsingenda-backend/src/Classifier/conf_checkpoint'
conf_data_base = '/root/work/Tsingenda-backend/src/Classifier/conf_datasets'

last_conf_cnt = {}

def mainloop():
    t = time.time()
    prev_t = time.time()
    last_data_cnt = AgendaData.objects.count()
    flag = True
    for user in User.objects.all():
        last_conf_cnt[user.name] = 0
    while True:
        if time.time() - t > update_int or flag:
            flag = False
            # update model
            data_cnt = AgendaData.objects.count()
            if data_cnt - last_data_cnt > update_data_inc:
                print('trigger main model update')
                # update main model
                # learn from scratch
                args = src_train.get_args(
                    save_dir=agenda_model_base,
                    use_db=True,
                    query_set=AgendaData.objects.all(),
                    init_from_ckpt=None
                    )
                model = src_train.get_model(args, 2)
                src_train.train(args, True, model)
                last_data_cnt = data_cnt
            # update confidence model
            for user in User.objects.all():
                if not user.django_user.is_authenticated: continue
                data_query = user.conf_data_set.all()
                username = user.name
                new_data_cnt = data_query.count()
                # stable dataset won't trigger re-training
                if new_data_cnt == last_conf_cnt[username]: continue
                print('trigger user [{}] confidence model update'.format(username))
                last_conf_cnt[username] = new_data_cnt
                model_path = user.conf_param.model_dict
                new_model_path = model_path
                if model_path is None:
                    new_model_path = os.path.join(conf_model_base, '{}_conf_model'.format(username))
                    user.conf_param = new_model_path
                    user.save()
                if data_query.count() == 0: continue
                # model = src_confidence.get_model(model_path)
                conf_path = user.conf_path
                # learn from scratch
                src_confidence.train(path=conf_path, data_query_set=data_query, model_path=None, new_model_path=new_model_path)
            t = time.time()
            
        if time.time() - prev_t > show_int:
            prev_t = time.time()
            print(time.time() - t, 's since last update')
            
if __name__ == '__main__':
    mainloop()