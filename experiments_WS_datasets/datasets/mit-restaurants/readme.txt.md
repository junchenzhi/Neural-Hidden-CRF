MIT-Restaurant Slot Fillings Dataset

# Source 
https://groups.csail.mit.edu/sls/downloads/restaurant/

# Entities
"Cuisine",
"Location",
"Hours",
"Dish",
"Amenity",
"Rating",
"Restaurant_Name",
"Price"

# Label Functions

16 labeling functions in total, as shown below

lfs = [
	"Location1",
    "Location2",
    "Cuisine",
    "Dish",
    "Price1",
    "Price2",
    "Amentity1",
    "Amentity2",
    "Rating1",
    "Rating2",
    "Restaurant_Name1",
    "Restaurant_Name2",
    "Hours1",
    "Hours2",
    "Hours3",
    "Yelp"
]

# Details on LFs

Location  ( |^)[^\w]*(within|near|next|close|nearby|around|around)[^\w]*([^\s]+ ){0,2}(here|city|miles|mile)*[^\w]*( |$)  any kid friendly restaurants around here 
Location  "in the area"  im looking for a 5 star restaurant in the area that serves wine 
Cuisine  cuisine1a=['italian','american','japanese','spanish','mexican','chinese','vietnamese','vegan']; cuisine1b=['bistro','delis']; cuisine2=['barbecue','halal','vegetarian','bakery']  can you find me some chinese food
Price  " ([0-9]+|few|under [0-9]+) dollar"  i need a family restaurant with meals under 10 dollars and kids eat  
Rating  "( (high|highly|good|best|top|well|highest|zagat) (rate|rating|rated))|((rated|rate|rating) [0-9]* star)|([0-9]+ star)"  where can i get the highest rated burger within ten miles 
Hours  "((open|opened) (now|late))|(still (open|opened|closed|close))|(((open|close|opened|closed) \w+([\s]| \w* | \w* \w* ))*[0-9]+ (am|pm|((a|p) m)|hours|hour))"  where is the nearest italian restaurant that is still open
Hours  "(open|close) (\w* ){0,3}until (\w* ){0,2}(([0-9]* (am|pm|((a|p) m)|hour|hours))|(late (night|hour))|(midnight))"  find a vegan cuisine which is open until 2 pm
Amenity  "(outdoor|indoor|group|romantic|family|outside|inside|fine|waterfront|outside|private|business|formal|casual|rooftop|(special occasion))([\s]| \w+ | \w+ \w+ )dining"  i want to go to a restaurant within 20 miles that got a high rating and is considered fine dining 
Hours  "(open |this |very ){0,2}late( night| dinner| lunch| dinning|( at night)){0,2}"  i need some late night chinese food within 4 miles of here 
Restaurant_Name  "[\w+ ]{0,2}(palace|cafe|bar|kitchen|outback|dominoes)"  is passims kitchen open at 2 am 
Dish  "wine|sandwich|pasta|burger|peroggis|burrito|(chicken tikka masala)|appetizer|pizza|winec|upcake|(onion ring)|tapas"  please find me a pub that serves burgers 
Price  "(affordable|cheap|expensive|inexpensive)"; bad_words=['the','a','an','has','have','that','this','beef','for','with','if','at']; 
  good_words=['price','prices','pricing','priced'];  im looking for an inexpensive mexican restaurant 
Rating  "(([0-9]*)|very|most)* (good|great|best|bad|excellent|negative|star) (\w* ){0,2}(review|reviews|rating|rated)"  which moderately priced mexican restaurants within 10 miles have the best reviews 
Amenity  "(pet|kid|)(friendly|indoor|outdoor|date|dining|buffet|great|fine|good|friend|group|birthday|anniversary|family|historical|family friendly|friendly)([\s]| \w+ | \w+ \w+ )(spot|dining|parking|dinne|style|eatries|catering|drive throughs|allow|amenity|amenity)*"  is there a pet friendly restaurant within 10 miles from here 
Restaurant_Name "(burger king|mcdonalds|taco bells|Mcdills|denneys|dennys|Mcdills)"  where is the next mcdonalds 

For Yelp: We extracted extract restaurant names and cuisines from yelp database (https://www.yelp.com/dataset) to augment the labeling function.


