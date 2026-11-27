from django.contrib.auth.backends import ModelBackend
from django.contrib.auth.models import User
from django.contrib.auth import get_user_model


class EmailBackend(ModelBackend):
    """
    Authenticate against settings.AUTH_USER_MODEL using case-insensitive email.
    """
    def authenticate(self, request, username=None, password=None, **kwargs):
        UserModel = get_user_model()
        email = kwargs.get('email', username)
        if not email or not password:
            return None

        try:
            # Case-insensitive lookup by email
            user = UserModel.objects.filter(email__iexact=email).first()
            if user and user.check_password(password) and self.user_can_authenticate(user):
                return user
        except UserModel.DoesNotExist:
            return None
        return None
