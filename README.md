# ApexRacing

_A short sentence capturing what the project is about_

---

## 1. Project Description  

**Rough Draft**  
The performance of motorsport vehicles is dictated by a range of key factors including braking, turning, and throttle application, each of which plays a critical role in determining speed and lap times. Split-second decisions by drivers at specific sections of the track can define overtaking opportunities, where even microseconds can change outcomes.  

Previous research on optimal driving strategies has often relied on simple simulations or narrow datasets. In this project, we analyze a **Formula One simulation** with a high-resolution dataset capturing vehicle behavior through Turns 1 and 2 at the Albert Park circuit in Melbourne. Our aim is to determine the optimum time until the vehicle reaches Turn 3 (TBD).  

The setup involves acquiring multiple reference datasets describing the race circuit and lap performance. We visualize the track, clean, and model the data, with the ultimate goal of uncovering insights into vehicle dynamics and racing strategy.  

---

## 2. Dataset  

### 2.1 Sources  

Our project uses several **Jupyter Notebooks** for analysis and visualization, and **Python files** for model training.  

Data sources include:  

- **UNSW F12024**: large-scale dataset of F1 racing lap performance  
- **f1sim-ref-left**: left track boundaries  
- **f1sim-ref-right**: right track boundaries  
- **f1sim-ref-turns**: corner and apex reference points  
- **f1sim-ref-line**: ideal racing line  

### 2.2 Data Description  

The goal of the final data product is to construct a dataset that captures the distinguishing features of driver performance through a cleaned variant of the original dataset, enhanced with feature-engineered variables. These features will allow us to redefine the current **f1sim-ref-line** optimal racing line (and associated trigger points) to accurately reflect the best possible path any driver can take that minimizes time over the sampled portion of the track (ending after Turn 3). The dataset will be designed so that a typical driver can follow this optimized line (changing braking, acceleration and steering upon a set of placed triggers) to achieve maximum speed through the first three turns.

Important dataset features are listed below:  

- **Observations (rows):** vehicle performance at each frame/step of the lap  
- **Features (columns):** speed, throttle, steering, braking, gear, RPM, world position, lap times, sector times, etc.  
- **Usage:** features will be engineered and aligned with track geometry to evaluate driver performance and racing lines.  

---

## 3. Workflow  

The workflow for this project:  

## Workflow  

### 3.1. Data acquisition  

We began by ingesting multiple reference raw datasets that describe the Albert Park circuit:

- The left and right track boundaries (_f1sim-ref-left.csv, f1sim-ref-right.csv_)
- A refrence racing line (_f1sim-ref-line.csv_)
- Apex point data and area for each corner (_f1sim-ref-turns_).

 These were combined with the UNSW F1 2024 lap telemetry dataset (UNSW F12024.csv), which contains detailed information about driver inputs (throttle, braking, steering) and car dynamics (speed, gear, RPM, position).  

### 3.2. Data and Cleaning

We began by cleaning the data, first removing unnecessary columns and renaming the remaining columns for clarity. The dataset also contained laps from tracks that were not part of our analysis (see figure ?).  

| ![Graph of amount of data by track](images/image.png) |
|:--:|
| _Figure ?. Number of data points per track, with approximate track ID layouts._ |

We therefore removed all data from tracks other than Albert Park, and further excluded datapoints beyond Turn 3, since our focus is on Turns 1 and 2 and the overtaking section between Turns 2 and 3. This was done by filtering for laps with a current lap distance below 1200m.  

Finally, we performed checks to handle missing values in the dataset, primarily for the car position variables. Rows with missing coordinates totaled around 68,000

Futher laps were removed as dicussed in the _Removing unsuitable laps_ section.

### Track visualisation  

To verify the data and provide context for later analysis, we reconstructed the circuit by plotting the left and right boundaries alongside the refrence racing line. Apex points were overlaid, and corners were annotated. We then produced zoomed-in visualisations of Turns 1 and 2, since these form the core section of interest.  

| ![Plot of track we are interested in](images/??) |
|:--:|
| _Figure ?. Number of data points per track, with approximate track ID layouts._ |

Should we add the sample turns?

### Removing unsuitable laps

- Removing rows with less than N (to be determined) data points so features could be constructed cleanly
- Removing laps where lap max distance between points become too great -> inaccuracy

### Feature engineering  

#### Track Width

We calculated the track width at each point along the circuit as a new feature. Using the reference datasets for the left and right track boundaries, we computed the Euclidean distance between corresponding points on each side of the track.  

This feature serves two purposes:  

1. It provides a spatial context for the car’s position along the track.  
2. It is used in off-track detection by comparing a car’s perpendicular distance from the track edges against the track width plus a buffer representing half the car’s width.  

![Track width](images/image-1.png)

#### Off track

To identify when cars went off track, we calculated each car’s perpendicular distance from both the left and right track boundaries and summed these distances. If the total distance exceeded the width of the track at that point, plus a buffer accounting for the car’s width, the car was considered off track.  

Since the provided coordinates do not account for the car’s physical width, we needed to choose an appropriate buffer. The dataset included an `INVALID_LAP` flag indicating whether the car went off track at any point in the lap. We used this flag to test different buffer values and selected the one that maximized the F1 score (Balancing false positives and false negatives) representing what is most likley the same buffer width used in the races / game simulation.

![alt text](images/image-2.png)

**(TO DO - CODE )**
We constructed new features to capture driver behaviour and vehicle dynamics more explicitly. These include braking and acceleration zones, steering angles, and measures of cornering precision. Each feature was designed as a separate transformation so that the pipeline can flexibly add or remove features depending on modelling needs.  

### Analysis and modelling (planned)  

Our next step is to evaluate driver performance through Turns 1–3, comparing how different inputs (braking, throttle, steering) impact lap time / speed / **(TO DO - THEORY)**.

---

## 4. Project Status  

### 4.1 Planning & Research  

- Conducted literature review on driver behavior, braking/throttle strategies, and racing simulations  
- Identified limitations in existing studies  

### 4.2 Dataset Construction  

- Interpreted dataset structure  
- Visualized variables and circuit geometry  
- Cleaned data (removed off-track laps, NaNs, and slowest drivers)  

### 4.3 Next Steps  

- Finalize feature engineering  
- Develop and test models for driver performance analysis  
- Evaluate results and compare across drivers  

---

## 5. Usage  

This product is intended for:  

- Analysis of driver performance in key track sections  
- Exploration of braking/acceleration patterns  
- Simulation of optimal racing strategies  

Future work may include:  

- Adding derived features (e.g., braking zones, acceleration windows)  
- Integrating external racing telemetry datasets  
- Building interactive simulations  

---

## 6. Contributors  

- **Data Transformation:** Charlotte Fang Hendro, Christian Joel, Eric Kim, Muhammad Ijaz, Samuel Katz  
- **Data Source:** Oracle – Stuart Coggins  
- **Guidance & Education:** Dr Jakub Stoklosa  

If you’d like to contribute:  

- Explore additional factors influencing lap time  
- Generate new features from existing data  
- Build simulations to test strategies  
- Integrate external datasets  

---

## 7. Support  

For questions or suggestions, contact:  

- Charlotte Fang Hendro – <z54_____@ad.unsw.edu.au>  
- Christian Joel – <z54_____@ad.unsw.edu.au>  
- Eric Kim – <z54_____@ad.unsw.edu.au>  
- Muhammad Ijaz – <z5417537@ad.unsw.edu.au>  
- Samuel Katz – <z5479193@ad.unsw.edu.au>  

---

## Assumptions  

- **Baseline assumptions:** derived from client consultation (Stuart, Oracle)
  - Current **f1sim-ref-line** ideal racing line is not indicative of fastest possible route for each driver (Stuart)  
- **Data cleaning assumptions:**  
  - Remove irrelevant tracks  
  - Remove NaNs in car coordinates  
  - Filter slower drivers (>75th percentile until Turn 3)  










