# Generated by Django 4.1.7 on 2023-03-30 08:01

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("dig_site", "0002_alter_workhistory_images_alter_workhistory_note_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="siteinfo",
            name="name",
            field=models.CharField(default="", max_length=128),
            preserve_default=False,
        ),
    ]
