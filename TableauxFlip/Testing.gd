extends VBoxContainer

func _ready():
	var test = {}
	
	test[[1, 2]] = "asdfc"
	
	print(test[[1, 3]])
