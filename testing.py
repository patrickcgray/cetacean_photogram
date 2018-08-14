import json
import skimage.draw


annotations = json.load(open("../../datasets/whale_training/blue/train/via_region_data.json"))


annotations = list(annotations.values())  # don't need the dict keys

# The VIA tool saves images in the JSON even if they don't have any
# annotations. Skip unannotated images.
annotations = [a for a in annotations if a['regions']]

a = annotations[0]

polygon_list = list(a['regions'].values())

print(polygon_list[0]['shape_attributes']['all_points_y'])


for i, p in enumerate(polygon_list):
	# Get indexes of pixels inside the polygon and set them to 1
	rr, cc = skimage.draw.polygon(p['shape_attributes']['all_points_y'], p['shape_attributes']['all_points_x'])
	if p['region_attributes']['body_part'] == "body":
		print(1)
	elif p['region_attributes']['body_part'] == "pectoral":
		print(2)
	

	#mask[rr, cc, i] = 1

