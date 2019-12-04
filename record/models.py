from django.db import models

# Create your models here.
class Face(models.Model):
    face_image = models.ImageField(upload_to='images/')
    name = models.CharField(max_length=150)
    face_id = models.IntegerField(primary_key=True)

    def __str__(self):
        return self.name

