# brownian brush

it does not use ai

it just turns the image into probabilities and then throws samples at a canvas

## what it does

lets you upload a png or jpg
turns it grayscale
samples more from darker parts of the image
can also use color sensitivity so colorful areas matter more too
can redraw it with dots
can redraw it with brownian looking lines
can use simple edge detection so outlines show up more
shows a little animation of the drawing appearing
lets you export the final image
lets you export the equations for the lines it drew
lets you export coordinates for dot modes

## why
i wanted a small math art project that looked cool but was actually simple

an image is basically just a grid of numbers so this treats the grid like a probability distribution

then the picture slowly shows up because the random samples keep landing in the important places

## how it works

first it converts the image to grayscale

then it treats black pixels as important and white pixels as not important

if

I(x,y) = brightness

where 0 is black and 1 is white, then

D(x,y) = 1 - I(x,y)

then it samples pixels using roughly

P(x,y) = D(x,y)^gamma / sum D(u,v)^gamma

so darker pixels get picked more often

for edge mode it also samples more from places where brightness changes quickly

for brownian lines it samples a target point and draws a short random walk near it

if color sensitivity is on, the dots and lines use colors from the original image instead of just one ink color

line equations

for brownian lines the app saves the equations for every tiny line segment it drew

it writes stuff like

endpoints: (x0, y0) -> (x1, y1)
parametric: x(t) = x0 + dx t, y(t) = y0 + dy t
cartesian: y = mx + b

if the line is vertical it writes

x = c

if color sensitivity is on it also saves the rgb color for each stroke

for dot modes it exports dot coordinates instead because there are no line equations

## how to use it

download or clone this

then run

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py

then open the local link it gives you

usually this is

http://127.0.0.1:5000

then upload an image and hit run

testing

run

pytest

privacy

nothing gets sent anywhere

it just runs locally and saves temporary output files on your computer

## note

this is not trying to be a real image editor


## todo maybe

make it faster for giant images
add better line styles
make the animation less janky
add a folder of examples
maybe make the ui less ugly
