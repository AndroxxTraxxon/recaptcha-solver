from captcha.image import ImageCaptcha
from string import ascii_lowercase, digits
import random
import os

symbols = ascii_lowercase + digits

image_gen = ImageCaptcha(fonts=['Oxygen-Sans-Bold.ttf'])

pwd = os.path.dirname(os.path.realpath(__file__))
num_images = 200000
while len(os.listdir(os.path.join(pwd, 'output'))) < num_images:
    text = ''.join(random.choices(symbols, k=5))
    data = image_gen.generate(text, format='png')
    with open(os.path.join(pwd, 'output', text + '.png'), 'wb+') as output:
        output.write(data.read())
    
