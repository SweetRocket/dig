# Generated by Django 4.1.7 on 2023-03-31 00:45

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("common", "0002_alter_profile_address_alter_profile_age_and_more"),
    ]

    operations = [
        migrations.AlterField(
            model_name="profile",
            name="emp",
            field=models.OneToOneField(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                to="common.employee",
            ),
        ),
    ]
