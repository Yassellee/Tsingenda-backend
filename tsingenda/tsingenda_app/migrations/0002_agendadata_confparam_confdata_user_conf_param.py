# Generated by Django 4.1.3 on 2022-12-10 08:04

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('tsingenda_app', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='AgendaData',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('raw_text', models.CharField(max_length=2000)),
                ('output', models.IntegerField()),
            ],
        ),
        migrations.CreateModel(
            name='ConfParam',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('model_dict', models.CharField(max_length=200)),
            ],
        ),
        migrations.CreateModel(
            name='ConfData',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('conf', models.FloatField()),
                ('output', models.IntegerField()),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='conf_data_set', to='tsingenda_app.user')),
            ],
        ),
        migrations.AddField(
            model_name='user',
            name='conf_param',
            field=models.OneToOneField(default=None, on_delete=django.db.models.deletion.CASCADE, to='tsingenda_app.confparam'),
        ),
    ]
