import os
from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from accounts.models import UserProfile, FavoriteCity


class Command(BaseCommand):
    help = "Creates or updates demo user demo@weatherapp.test with password from DEMO_PASSWORD env var"

    def handle(self, *args, **options):
        password = os.environ.get('DEMO_PASSWORD', '').strip()
        if not password:
            self.stdout.write(self.style.WARNING("DEMO_PASSWORD environment variable is not set. Skipping demo user creation."))
            return

        email = "demo@weatherapp.test"
        user, created = User.objects.get_or_create(
            email=email,
            defaults={
                'username': email,
                'first_name': 'Demo',
                'last_name': 'User',
            }
        )
        user.set_password(password)
        user.save()

        # Ensure profile defaults
        profile, _ = UserProfile.objects.get_or_create(user=user)
        if not profile.default_city:
            profile.default_city = 'London'
            profile.units = 'C'
            profile.save()

        # Add starter favorites if empty
        if not FavoriteCity.objects.filter(user=user).exists():
            FavoriteCity.objects.create(user=user, city='London', country='GB')
            FavoriteCity.objects.create(user=user, city='Mumbai', country='IN')

        action = "Created" if created else "Updated"
        self.stdout.write(self.style.SUCCESS(f"{action} demo user: {email}"))
