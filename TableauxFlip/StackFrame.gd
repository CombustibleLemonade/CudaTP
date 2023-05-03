extends Control

var substitutions = {}
var substitution_idx = 0

var clause
var literal_idx

var unifies = false

var literal_labels = []

func _ready():
	for literal in clause.literal_list:
		if get_child_count() != 0:
			var label = Label.new()
			label.text = " | "
			add_child(label)
		
		var literal_label = Label.new()
		literal_labels.append(literal_label)
		add_child(literal_label)
	
	update_sub()

func update_sub(can_unify = false):
	for i in range(len(literal_labels)):
		if i == literal_idx:
			if can_unify:
				literal_labels[i].modulate = Color.GREEN
			else:
				literal_labels[i].modulate = Color.AQUA
		else:
			literal_labels[i].modulate = Color.WHITE
		
		literal_labels[i].text = clause.literal_list[i].to_string_with_substitutions(substitutions, substitution_idx)
