import os
import shutil
import tempfile
from unittest.mock import Mock, patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse

from accounts.models import User
from translater.models import Video


class UploadVideoTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            email="translator@example.com",
            password="StrongPassword123!",
            is_active=True,
            available_seconds=60,
        )
        self.client.force_login(self.user)
        self.output_dir = tempfile.mkdtemp(prefix="videolingo-upload-test-")
        self.uploads_dir = tempfile.mkdtemp(prefix="videolingo-source-test-")
        self.addCleanup(lambda: shutil.rmtree(self.output_dir, ignore_errors=True))
        self.addCleanup(lambda: shutil.rmtree(self.uploads_dir, ignore_errors=True))

    @patch("translater.views.get_video_duration_seconds", return_value=125)
    def test_upload_video_shows_topup_cta_when_minutes_are_insufficient(self, _mock_duration):
        with override_settings(OUTPUT_DIR=self.output_dir, UPLOADS_DIR=self.uploads_dir):
            response = self.client.post(
                reverse("upload_video"),
                {
                    "file": SimpleUploadedFile("clip.mp4", b"fake video bytes", content_type="video/mp4"),
                    "language": "en",
                    "volume": "12",
                    "is_sub": "on",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Недостаточно доступных минут для запуска перевода.")
        self.assertContains(response, "Перейти к оплате")
        self.assertContains(response, reverse("billing_dashboard"))
        self.assertContains(response, "нужно 3 мин., сейчас доступно 1 мин.", html=False)
        self.assertEqual(Video.objects.count(), 0)

    @patch("translater.views.requests.post")
    @patch("translater.views.get_video_duration_seconds", return_value=60)
    def test_upload_video_enqueues_task_with_source_path_in_uploads_dir(self, _mock_duration, mock_post):
        self.user.available_seconds = 600
        self.user.save(update_fields=["available_seconds"])

        mock_response = Mock()
        mock_response.json.return_value = {"task_id": "celery-task-123"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with override_settings(OUTPUT_DIR=self.output_dir, UPLOADS_DIR=self.uploads_dir):
            response = self.client.post(
                reverse("upload_video"),
                {
                    "file": SimpleUploadedFile("clip.mp4", b"fake video bytes", content_type="video/mp4"),
                    "language": "en",
                    "volume": "12",
                },
            )

        video = Video.objects.get()
        self.assertRedirects(response, reverse("translate_status", kwargs={"video_id": video.pk}))
        self.assertEqual(video.status, "PENDING")
        self.assertEqual(video.task_id, "celery-task-123")

        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(payload["save_dir"], f"users/{self.user.pk}/videos/{video.pk}")
        self.assertTrue(payload["source_path"].startswith(self.uploads_dir))
        self.assertTrue(os.path.exists(payload["source_path"]))
        self.assertFalse(payload["source_path"].startswith(self.output_dir))

    def test_upload_video_rejects_non_video_file(self):
        with override_settings(OUTPUT_DIR=self.output_dir, UPLOADS_DIR=self.uploads_dir):
            response = self.client.post(
                reverse("upload_video"),
                {
                    "file": SimpleUploadedFile("payload.exe", b"fake bytes", content_type="application/octet-stream"),
                    "language": "en",
                    "volume": "12",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Допустимы только видеофайлы.")
        self.assertEqual(Video.objects.count(), 0)

    def test_upload_video_rejects_oversized_file(self):
        oversized = SimpleUploadedFile("clip.mp4", b"123456", content_type="video/mp4")
        oversized.size = 6

        with override_settings(
            OUTPUT_DIR=self.output_dir,
            UPLOADS_DIR=self.uploads_dir,
            MAX_VIDEO_UPLOAD_SIZE_MB=0,
            MAX_VIDEO_UPLOAD_SIZE_BYTES=5,
        ):
            response = self.client.post(
                reverse("upload_video"),
                {
                    "file": oversized,
                    "language": "en",
                    "volume": "12",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Размер файла превышает лимит")
        self.assertEqual(Video.objects.count(), 0)
