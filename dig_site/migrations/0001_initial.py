# Generated by Django 4.1.7 on 2023-03-30 07:26

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ("common", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="SiteInfo",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("address", models.TextField()),
                ("relic_era", models.TextField()),
                ("area", models.FloatField()),
            ],
        ),
        migrations.CreateModel(
            name="WorkHistory",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("zone", models.CharField(max_length=128)),
                ("note", models.TextField()),
                ("date", models.DateField()),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("images", models.ManyToManyField(to="common.image")),
                (
                    "site",
                    models.ForeignKey(
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        to="dig_site.siteinfo",
                    ),
                ),
                ("workers", models.ManyToManyField(to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name="Report",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "age",
                    models.PositiveIntegerField(
                        choices=[(0, "구석기"), (1, "신석기"), (2, "청동기"), (3, "철기")]
                    ),
                ),
                (
                    "region",
                    models.PositiveIntegerField(
                        choices=[
                            (0, "서울"),
                            (1, "경기도"),
                            (2, "강원도"),
                            (3, "충청북도"),
                            (4, "충청남도"),
                            (5, "경상북도"),
                            (6, "경상남도"),
                            (7, "전라북도"),
                            (8, "전라남도"),
                            (9, "제주도"),
                        ]
                    ),
                ),
                ("name", models.CharField(max_length=128)),
                ("url", models.URLField()),
                (
                    "image",
                    models.ForeignKey(
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        to="common.image",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="SiteJoin",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "site",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="dig_site.siteinfo",
                    ),
                ),
                (
                    "user",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                "verbose_name_plural": "SiteJoin",
                "unique_together": {("user", "site")},
                "index_together": {("user", "site")},
            },
        ),
    ]
