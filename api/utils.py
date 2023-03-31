from dig_site.models import WorkHistory


def work_to_dict(work: WorkHistory):
    site = work.site
    if site is None:
        site = -1
    else:
        site = site.pk
    
    return {
        'id': work.pk,
        'zone': work.zone,
        'date': work.date.strftime('%Y-%m-%d'),
        'images': [i.image.url for i in work.images.all()],
        'note': work.note,
        'workers': [w.pk for w in work.workers.all()],
        'site': site,
        'site_name': work.site.name if work.site is not None else '',
        'updated_at': work.updated_at.strftime('%Y-%m-%d %H:%M:%S'),
        'updated_at_timeonly': work.updated_at.strftime('%H:%M:%S')
    }
