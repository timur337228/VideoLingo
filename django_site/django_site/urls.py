"""
URL configuration for django_site project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/6.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.contrib.sitemaps.views import sitemap
from django.urls import include, path
from django.views.generic import TemplateView

from .sitemaps import LanguagePairSitemap, PublicPagesSitemap
from .views import (
    contacts,
    language_pair,
    offer,
    payments_refunds,
    preview,
    privacy_policy,
    robots_txt,
)


sitemaps = {
    "pages": PublicPagesSitemap,
    "language-pairs": LanguagePairSitemap,
}

urlpatterns = [
    path("", preview, name="home"),
    path("preview/", preview, name="preview"),
    path(
        "google62cd21c736d230fd.html",
        TemplateView.as_view(
            template_name="google62cd21c736d230fd.html",
            content_type="text/html; charset=utf-8",
        ),
    ),
    path(
        "yandex_7cd360ccdc45f960.html",
        TemplateView.as_view(
            template_name="yandex_7cd360ccdc45f960.html",
            content_type="text/html; charset=utf-8",
        ),
    ),
    path("robots.txt", robots_txt, name="robots_txt"),
    path("sitemap.xml", sitemap, {"sitemaps": sitemaps}, name="django.contrib.sitemaps.views.sitemap"),
    path(
        "translate-video/<slug:source_slug>-to-<slug:target_slug>/",
        language_pair,
        name="language_pair",
    ),
    path("offer/", offer, name="offer"),
    path("privacy-policy/", privacy_policy, name="privacy_policy"),
    path("contacts/", contacts, name="contacts"),
    path("payments-refunds/", payments_refunds, name="payments_refunds"),
    path("admin/", admin.site.urls),
    path("auth/", include("accounts.urls")),
    path("billing/", include("billing.urls")),
    path("translate/", include("translater.urls")),
    path("social-auth/", include("social_django.urls", namespace="social")),

]
