#Nastavenia projektu. Pri zmene tried stačí upraviť tento súbor

CLASSES = ["eat", "clap"]

KINETICS_CLASSES = {
    "eat": ["dining", "tasting food", "eating burger", "eating cake", "eating carrots",
            "eating chips", "eating doughnuts", "eating hotdog", "eating ice cream",
            "eating spaghetti", "eating watermelon"],
    "drink": ["drinking", "drinking beer", "drinking shots", "tasting beer"],
    "clap": ["clapping", "applauding"],
}

DATA_DIR = "HMDB51"        
INDEX_FILE = "index.csv"  
RESULTS_DIR = "results"    

N_FRAMES = 16             
SIZE = 224                 

N_SPLITS = 10              # počet rôznych delení na trénovacie a testovacie dáta
TEST_SIZE = 0.3            # podiel testovacích dát
