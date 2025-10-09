﻿# ApexRacing

**A data-driven analysis of telemetry-based performance in F1 simulation racing, investigating optimal behavioural car metrics through Turns 1 and 2 of Melbourne Albert Park circuit.**

---

## 1. Project Description  

Performances in motorsports are dictated by a range of key factors regarding the braking, turning and throttle of the vehicle, where each plays a critical role in determining the final lap time. Often shaped by the split-second decisions made by drivers at specific sections of the track, turns are one such crucial point as they present opportunities for overtaking. As even microsecond differences can dictate differences in these outcomes, minimising the time required to pass through these zones becomes essential, especially in competitive contexts such as Formula One races.

Previous research on optimal driving strategies has often relied on simple simulations or narrow datasets. In this project, we analyse a **Formula One simulation** with a high-resolution dataset capturing vehicle behaviour through Turns 1 and 2 at the Melbourne Albert Park circuit. Our primary aim is to determine the optimum behavioural car conditions (i.e. braking, throttle, steering) at which the vehicle enters Turn 3, based on the time to reach a fixed distance point.

The setup involves acquiring multiple reference datasets describing the race circuit and lap performance. Our proposed final dataframe integrates multiple reference data points, capturing key performance indicators (braking point, throttle point, etc) alongside aggregated measures (total throttle, etc) to further complement the evaluation of performances. These tasks highlight a significant initial step in bridging the gap between model driven optimisation and real world telemetry analysis in the modern motorsport landscape.

---

## 2. Dataset  

### 2.1 Sources  

Our project uses several **Jupyter Notebooks** for analysis and visualization, and **Python files** for building the complete dataset.  

Data sources include:  

- **UNSW F12024**: large-scale dataset of F1 racing lap performance  
- **f1sim-ref-left**: left track boundaries  
- **f1sim-ref-right**: right track boundaries  
- **f1sim-ref-turns**: corner and apex reference points  
- **f1sim-ref-line**: ideal racing line  

### 2.2 Data Description  

The goal of the final data product is to construct a dataset that captures the distinguishing features of driver performance through a cleaned variant of the original dataset, enhanced with feature-engineered variables. These features will allow us to redefine the current **f1sim-ref-line** optimal racing line (and associated trigger points) to accurately reflect the best possible path any driver can take that minimises time over the sampled portion of the track (ending after Turn 3). The dataset will be designed so that a typical driver can follow this optimised line (changing braking, acceleration and steering upon a set of placed triggers) to achieve maximum speed through the first three turns.

Important dataset features are listed below:  

- **Observations (rows):** vehicle performance at each frame/step of the lap  
- **Features (columns):** speed, throttle, steering, braking, gear, RPM, world position, lap times, sector times, etc.  
- **Usage:** features will be engineered and aligned with track geometry to evaluate driver performance and racing lines.  

| Description                | Variables                                                                                                                                | Features               | Usage                                                                                    |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- | ---------------------------------------------------------------------------------------- |
| **Session Identifiers**       | `SESSION_GUID`, `M_SESSIONUID`, `R_SESSION`, `R_NAME`, `R_STATUS`, `M_TRACKID`                                                           | Categorical/Text       | Groups laps/runs; links telemetry to results/metadata; filters by track/session context |
| **Lap Data**                  | `M_CURRENTLAPNUM`, `M_CURRENTLAPNUM_1`, `M_CURRENTLAPINVALID_1`, `LAPTIME`, `CURRENTLAPTIME`, `M_CURRENTLAPTIMEINMS_1`                   | Discrete/Binary/Time   | Lap segmentation, filtering invalid laps, and benchmarking performance                  |
| **Driver Inputs**             | `M_THROTTLE_1`, `M_BRAKE_1`, `M_STEER_1`, `M_GEAR_1`, `M_FRONTWHEELSANGLE`, `M_DRS_1`                                                    | Continuous/Categorical | Evaluates acceleration, braking, cornering, gear shifts, and DRS use                    |
| **Speed & Engine**            | `M_SPEED_1`, `M_ENGINERPM_1`                                                                                                             | Continuous             | Core performance measures; assess acceleration, straight-line speed, and shift points   |
| **Brake Temperatures**        | `M_BRAKESTEMPERATURE_FL_1`, `M_BRAKESTEMPERATURE_FR_1`, `M_BRAKESTEMPERATURE_RL_1`, `M_BRAKESTEMPERATURE_RR_1`                           | Continuous (°C)        | Thermal load and braking efficiency; detects imbalance/overheating                      |
| **Tyre Pressures**            | `M_TYRESPRESSURE_FL_1`, `M_TYRESPRESSURE_FR_1`, `M_TYRESPRESSURE_RL_1`, `M_TYRESPRESSURE_RR_1`                                           | Continuous (psi/kPa)   | Grip and stability; monitors balance and traction                                       |
| **Distances**                 | `M_LAPDISTANCE_1`, `M_TOTALDISTANCE_1`                                                                                                   | Continuous (m)         | Aligns telemetry with track position; cumulative mileage tracking                       |
| **World Position (Car)**      | `M_WORLDPOSITIONX_1`, `M_WORLDPOSITIONY_1`, `M_WORLDPOSITIONZ_1`                                                                         | Continuous             | 3D trajectory mapping for racing line, elevation, and kerbs                             |
| **World Orientation Vectors** | `M_WORLDFORWARDDIRX_1`, `M_WORLDFORWARDDIRY_1`, `M_WORLDFORWARDDIRZ_1`, `M_WORLDRIGHTDIRX_1`, `M_WORLDRIGHTDIRY_1`, `M_WORLDRIGHTDIRZ_1` | Continuous             | Car heading/orientation in 3D space; used in angle-to-apex, yaw/roll calculations       |
| **Car Angles**                | `M_YAW_1`, `M_PITCH_1`, `M_ROLL_1`                                                                                                       | Continuous (degrees)   | Captures rotation dynamics - heading, dive/squat, and body roll                         |
| **Track Reference Data**      | `FRAME`, `WORLDPOSX`, `WORLDPOSY`, `APEX_X1`, `APEX_Y1`, `CORNER_X1…Y2`, `TURN`                                                          | Mixed                  | Defines track geometry, apex points, corners, and reference frames                      |
| **Engineered Features**       | `dist_apex_1`, `dist_apex_2`, `angle_to_apex1`, `angle_to_apex2`, `track_width`, `left_dist`, `right_dist`, `l_width`, `r_width`, `in`   | Continuous/Binary      | Derived metrics for racing line, corner approach, and track usage evaluation            |


### 2.3 Assumptions  

- **Baseline assumptions:** derived from client consultation (Stuart, Oracle)
  - Current **f1sim-ref-line** ideal racing line is not indicative of fastest possible route for each driver (Stuart)  
- **Data cleaning assumptions:**  
  - Remove irrelevant tracks  
  - Remove NaNs in car coordinates  
  - Filter slower drivers (>75th percentile until Turn 3)  

---

## 3. Workflow  

The workflow for this project:  

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
| _Figure 1. Number of data points per track, with approximate track ID layouts._ |

We therefore removed all data from tracks other than Albert Park, and further excluded datapoints beyond Turn 3, since our focus is on Turns 1 and 2 and the overtaking section between Turns 2 and 3. This was done by filtering for laps with a current lap distance below 1200m.  

Finally, we performed checks to handle missing values in the dataset, primarily for the car position variables. Rows with missing coordinates totaled around 68,000

Futher laps were removed as dicussed in the _Removing unsuitable laps_ section.

### 3.3. Track visualisation  

To verify the data and provide context for later analysis, we reconstructed the circuit by plotting the left and right boundaries alongside the refrence racing line. Apex points were overlaid, and corners were annotated. We then produced zoomed-in visualisations of Turns 1 and 2, since these form the core section of interest.  

| ![Plot of track we are interested in](images/??) |
|:--:|
| _Figure 2. Number of data points per track, with approximate track ID layouts._ |

Should we add the sample turns?

### 3.4. Removing unsuitable laps

- Removing rows with less than N (to be determined) data points so features could be constructed cleanly
- Removing laps where lap max distance between points become too great -> inaccuracy

### 3.5. Feature Engineering  

#### Track Width

We calculated the track width at each point along the circuit as a new feature. Using the reference datasets for the left and right track boundaries, we computed the Euclidean distance between corresponding points on each side of the track.  

This feature serves two purposes:  

1. It provides a spatial context for the car’s position along the track.  
2. It is used in off-track detection by comparing a car’s perpendicular distance from the track edges against the track width plus a buffer representing half the car’s width.  

![Track width](images/image-1.png)

#### Off track

To identify when cars went off track, we calculated each car’s perpendicular distance from both the left and right track boundaries and summed these distances. If the total distance exceeded the width of the track at that point, plus a buffer accounting for the car’s width, the car was considered off track.  

Since the provided coordinates do not account for the car’s physical width, we needed to choose an appropriate buffer. The dataset included an `INVALID_LAP` flag indicating whether the car went off track at any point in the lap. We used this flag to test different buffer values and selected the one that maximised the F1 score (Balancing false positives and false negatives) representing what is most likley the same buffer width used in the races / game simulation.

![alt text](images/image-2.png)

**(TO DO - CODE )**

We constructed new features to capture driver behaviour and vehicle dynamics more explicitly. These include braking and acceleration zones, steering angles, and measures of cornering precision. Each feature was designed as a separate transformation so that the pipeline can flexibly add or remove features depending on modelling needs.  

### 3.6. Feature Documentation

The following documentation expands on our engineered features by defining two key layers of our feature design - **Moments** and **Attrbutes** - and illustrate how these interact in our dataframe.

#### Moments

Each **Moment** represents a key behavioural event that occurs during a lap. For instance, when a driver first applies the brake, releases the throttle, or reaches the midpoint of a turn. These reference points are used to anchor subsequent calculations to measure timing, distance, and performance changes through each section of the track.

| **Type of Moment** | **Moment** | **Code** | **Time-to-Extrema** | **Description** |
|--------------------:|------------------------:|-----------------------:|--------------------:|----------------:|
| Variable | First Brake | `BP` | Yes | Captures the point at which braking is first initiated before Turn 1 or 2 |
| Variable | End Brake | `brake_end` | Yes | Marks the release of braking input |
| Variable | Start Steering | `SS1`, `SS2` | ? | Identifies the first notable steering input, signalling the driver’s approach to turn-in |
| Variable | End Steering | `ES1`, `ES2` | ? | Captures the point where steering angle returns to neutral after a turn |
| Variable | Middle Turning Point | `TM` | No | Represents the midpoint of steering angle |
| Variable | Off Throttle | `ET1` | Yes | Indicates when the driver fully releases throttle before entering a corner |
| Variable | Start Throttle | `FT1` | Yes | Marks the moment throttle is reapplied after corner exit |
| Fixed | Apex (actual) | `APEX1`, `APEX2` | No | Defines the true geometric apex points of Turns 1 and 2 |
| Fixed | Distances | `CP360`, `CP430`, `CP530`, `CP900` | No | Reference points from start line |

Overall, these **Moments** define the critical phases of vehicle behaviour during Turns 1–3 and are used as anchor points for deriving further measurements.

---

#### Attributes

**Attributes** describe *what* is being measured at or around each Moment. They capture the car’s physical and temporal states (position, distance, steering angle) to quantify how the driver’s input changes through each phase.

| **Attribute** | **Code Pattern / Suffix** | **Type / Shape** | **Description** |
|----------------|---------------------------|------------------|-----------------|
| Lap distance | `*_LD` | Scalar (m) | Linear distance travelled at each moment |
| Position X/Y | `*_X`, `*_Y` | Scalar | Car coordinates used to reconstruct trajectories |
| Timestamp / Lap time | `*_T` | Scalar | Captures temporal alignment of events |
| Relative displacement | `*_R` | Scalar (m) | Perpendicular offset from racing line or track edge |
| Distance to apex | `*_APEX1_D`, `*_APEX2_D` | Scalar (m) | Euclidean distance from car position to turn apex |
| Angle to apex | `*_APEX1_A`, `*_APEX2_A` | Angle | Angular offset between car heading and apex vector |
| Steering angle | `*_STEER` | Angle | Direction and intensity of steering input |
| Velocity vs tire direction | `*_VEL_ANG` | Angle | Difference between velocity vector and tyre direction |
| Time to extrema | `*_TE` | Scalar | Time difference between current frame and the feature’s peak event |
| Rotational forces | `*_PITCH`, `*_YAW`, `*_ROLL` | Scalar | Captures vehicle rotation around each axis to analyse corner dynamics |

---

#### How to Read Documentation  

The **Moments** and **Attributes** tables are designed to be read together. Each feature in the engineered dataset is formed by combining a **Moment code** (indicating *when* the measurement occurs) with an **Attribute suffix** (indicating *what* is being measured).  

For example:

- **`BP_STEER`** - steering angle recorded at the **First Brake** moment
- **`FT1_APEX1_D`** - distance from the car to the first apex when **Start Throttle** occurs
- **`TM_YAW`** - yaw rotation (rate of directional change) measured at the **Middle Turning Point** 
- **`ET1_TE`** - time to peak deceleration from the **Off Throttle** event

This modular naming convention ensures every feature is able to be interpreted and traced back to its functional purpose within the lap.

By following this convention, users may efficiently locate, filter, and compare driver performance metrics across moments, laps, and turns, enabling consistent and replicable analysis across future model iterations.

### Analysis and modelling (planned)  

Our next step is to evaluate driver performance through Turns 1–3, comparing how different inputs (braking, throttle, steering) impact lap time / speed / **(TO DO - THEORY)**.

---

## 4. Project Status  

### 4.1 Planning & Research  

- Conducted literature review on driver behavior, braking/throttle strategies, and racing simulations  
- Identified limitations in existing studies  

### 4.2 Dataset Construction  

- Interpreted dataset structure  
- Visualised variables and circuit geometry  
- Cleaned data (removed off-track laps, NaNs, and slowest drivers)  

### 4.3 Next Steps  

- Finalise feature engineering  
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

- Charlotte Fang Hendro – <z5363431@ad.unsw.edu.au>  
- Christian Joel – <z5257354@ad.unsw.edu.au>  
- Eric Kim – <z5478624@ad.unsw.edu.au>  
- Muhammad Ijaz – <z5417537@ad.unsw.edu.au>  
- Samuel Katz – <z5479193@ad.unsw.edu.au>  

---





