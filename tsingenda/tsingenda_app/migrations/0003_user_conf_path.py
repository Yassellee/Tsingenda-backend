# Generated by Django 4.1.3 on 2022-12-11 12:08

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('tsingenda_app', '0002_agendadata_confparam_confdata_user_conf_param'),
    ]

    operations = [
        migrations.AddField(
            model_name='user',
            name='conf_path',
            field=models.CharField(default='', max_length=200),
            preserve_default=False,
        ),
    ]
