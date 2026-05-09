```mermaid
flowchart TD
	node1["data_ingestion"]
	node2["data_preprocessing"]
	node3["feature_eng"]
	node1-->node2
	node2-->node3
	node4["model_building"]
	node5["model_evaluation"]
	node6["model_registration"]
	node4-->node5
	node5-->node6
```