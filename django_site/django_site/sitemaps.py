from django.contrib.sitemaps import Sitemap
from django.urls import reverse

from .languages import get_language_pairs


class PublicPagesSitemap(Sitemap):
    protocol = "https"
    priority_map = {
        "home": 1.0,
        "offer": 0.4,
        "privacy_policy": 0.3,
        "contacts": 0.5,
        "payments_refunds": 0.6,
    }
    changefreq_map = {
        "home": "weekly",
        "offer": "monthly",
        "privacy_policy": "monthly",
        "contacts": "monthly",
        "payments_refunds": "weekly",
    }

    def items(self):
        return [
            "home",
            "offer",
            "privacy_policy",
            "contacts",
            "payments_refunds",
        ]

    def location(self, item):
        return reverse(item)

    def priority(self, item):
        return self.priority_map.get(item, 0.5)

    def changefreq(self, item):
        return self.changefreq_map.get(item, "weekly")


class LanguagePairSitemap(Sitemap):
    protocol = "https"
    changefreq = "weekly"
    priority = 0.8

    def items(self):
        return get_language_pairs()

    def location(self, item):
        return reverse(
            "language_pair",
            kwargs={
                "source_slug": item["source"]["slug"],
                "target_slug": item["target"]["slug"],
            },
        )
