#=
Modeling Stroke Patient Triage and Transport in CA with MDPs
Autumn 2023 CS199/CS195 Project
Emily Molins and Yasmine Alonso

File: STPMDP.jl
---------------
This file contains the implementation of the MDP for the triage/transport problem.
=#

using Pkg
using CSV
using DataFrames
using DelimitedFiles
using D3Trees
using DiscreteValueIteration
using Graphs
using LaTeXStrings
using LinearAlgebra
using LocalApproximationValueIteration
using MCTS
using POMDPModels
using POMDPTools
using POMDPs
using Parameters
using Plots
using Printf
using PyCall
using QuickPOMDPs
using QMDP
using Random
using RollingFunctions
using SpecialFunctions
using Statistics
using TabularTDLearning
using LinearAlgebra
using HTTP
using JSON


@enum LocType FIELD CLINIC PSC CSC
@enum StrokeTypeKnown UNKNOWN KNOWN
@enum StrokeType LVO NLVO HEMORRHAGIC MIMIC

# File from which we will read in all the information about hospital metrics, location, etc
hospital_info_file = "HospitalFiles/notgoodhospitals.csv"

mutable struct Location
    name::String  # i.e. "STANFORD"
    latlon::Tuple{Float64, Float64}  # Location of the hospital
    performance_metric::Float64  # transfer time if CLINIC/PSC/CSC, -1 otherwise
    type::LocType  # FIELD CLINIC PSC CSC
end

# Define PatientState Struct
struct PatientState
    loc::Location # Current location of patient, represented as Location struct
    t_onset::Float64 # Keeps track of time from onset to now
    stroke_type_known::StrokeTypeKnown  # UNKNOWN or KNOWN based on whether we know 
    stroke_type::StrokeType   
end

#@enum Action ROUTE_STANFORD ROUTE_REGIONAL ROUTE_GOODSAMARITAN ROUTE_UCSF ROUTE_KAISER ROUTE_ELCAMINO
@enum Action begin
    ROUTE_ChineseHospital
    ROUTE_CPMCDaviesCampus
    ROUTE_CPMCMissionBernalCampus
    ROUTE_CPMCVanNessCampus
    ROUTE_KaiserPermanenteSanFranciscoMedicalCenter
    ROUTE_KaiserPermanenteSouthSanFranciscoMedicalCenter
    ROUTE_LagunaHondaHospitalandRehabilitationCenter
    ROUTE_SaintFrancisMemorialHospital
    ROUTE_SaintMarysMedicalCenter
    ROUTE_SanFranciscoVAMedicalCenter
    ROUTE_UCSFBettyIreneMooreWomensHospital
    ROUTE_UCSFHelenDillerMedicalCenteratParnassusHeights
    ROUTE_UCSFMedicalCenteratMissionBay
    ROUTE_ZuckerbergSanFranciscoGeneralHospitalandTraumaCenter
    ROUTE_AltaBatesSummitMedicalCenter
    ROUTE_OaklandMedicalCenter
    ROUTE_WilmaChanHighlandHospital
    ROUTE_SantaClaraMedicalCenter
    ROUTE_SantaClaraValleyMedicalCenter
    ROUTE_OConnorHospital
    ROUTE_RegionalMedicalCenterofSanJose
    ROUTE_GoodSamaritanHospital
    ROUTE_PaloAltoVAMedicalCenter
    ROUTE_StanfordHospital
    ROUTE_AlamedaHospital
    ROUTE_ProvidenceQueenoftheValleyMedicalCenter
    ROUTE_AntiochMedicalCenter
    ROUTE_SutterDeltaMedicalCenter
    ROUTE_SanMateoMedicalCenter
    ROUTE_MarinHealthMedicalCenter
    ROUTE_WashingtonHospitalHealthcareSystem
    ROUTE_WalnutCreekMedicalCenter
    ROUTE_FairmontRehabilitationandWellness
    ROUTE_ProvidenceSantaRosaMemorialHospital
    ROUTE_SantaRosaHospital
    ROUTE_SutterSantaRosaRegionalHospital
    ROUTE_SonomaValleyHospital
    ROUTE_KaiserPermanenteRedwoodCity
    STAY
end



# Converts a string into its Action representation (or any other enum)
# i.e. converts "ROUTE_STANFORD" to ROUTE_STANFORD
string_to_enum(str) = eval(Meta.parse(str))

# Converts an action into its string representation
# i.e. converts ROUTE_STANFORD to "ROUTE_STANFORD"
function enum_to_string(action)
    return(String(Symbol(action)))
end

# In: a CSV file representing hospitals
# Out: a vector of Locations
function csv_to_locations(file)
    df = CSV.read(file, DataFrame, delim=',')
    locs = []
    for row in eachrow(df)
        hospital = row["Hospital"]
        lat = row["Lat"]
        lon = row["Lon"]
        tup = (lat, lon)
        metric = float(row["Performance Metric"])
        type = string_to_enum(row["Type"])
        push!(locs, Location(hospital, tup, metric, type))
    end
    return locs
end


# Custom MDP type
@with_kw struct StrokeMDP <: MDP{PatientState, Action}
    # Defined all constants within this StrokeMDP struct--now, we can access all fields whenever we have an instance of the MDP
    
    p_LVO = 0.4538  # Probability of a large vessel occlusion
    p_nLVO = 0.1092  # Probability of a non-large vessel occlusion
    p_Hemorrhagic = 0.3445  # probability of a hemorrhagic stroke
    p_Mimic = 0.0924  # Probability of a stroke mimic

    #! Pull from CSV file
    locations::Vector{Location} = csv_to_locations(hospital_info_file)
        # [Location("STANFORD", (37.44170119292893, -122.16935983265613), 30, CSC), 
        #     Location("REGIONAL", (37.36284947319352, -121.84981030382419), 30, CSC),
        #     Location("GOODSAMARITAN", (37.251918539458636, -121.94665001054544), 30, PSC),
        #     Location("UCSF", (37.762947142641124, -122.45801390380367), 30, CSC),
        #     Location("KAISER", (37.336041307978924, -121.99947229909202), 45, PSC),
        #     Location("ELCAMINO", (37.3692428362012, -122.07954931385208), 45, PSC),
        #     Location("FIELD1", (37.330552, -122.213429), -1, FIELD)
        # ] 

    γ = 0.95  # Discount factor 
    #API_KEY = ""  

end

POMDPs.discount(m::StrokeMDP) = m.γ

# loc::Location # Current location of patient, represented as Location struct
# t_onset::Float64 # Keeps track of time from onset to now
# stroke_type_known::StrokeTypeKnown  # UNKNOWN or KNOWN based on whether we know 
# stroke_type::StrokeType  
function POMDPs.states(m::StrokeMDP)
    𝒮 = Vector{PatientState}()
    for loc in m.locations
        for t_onset in 0:720
            for known in [UNKNOWN, KNOWN]
                for stroke_type in [LVO, NLVO, HEMORRHAGIC, MIMIC]
                    push!(𝒮, PatientState(loc, t_onset, known, stroke_type))
                end
            end
        end
    end
    
    return 𝒮
end

# Action function! 
function POMDPs.actions(m::StrokeMDP, s::PatientState)
    if s.loc.type == FIELD
        lst = ["ROUTE_$(hospital.name)" for (hospital) in m.locations if hospital.type != FIELD]
        return lst
      
    # If patient has already been routed to a Clinic, can be transferred to either a PSC or CSC
    elseif s.loc.type == CLINIC
        lst = ["ROUTE_$(hospital.name)" for (hospital) in m.locations if hospital.type == PSC || hospital.type == CSC]
        push!(lst, "STAY")
        return lst
    
    # If patient has already been routed to a PSC, can be transferred to a CSC
    elseif s.loc.type == PSC
        lst = ["ROUTE_$(hospital.name)" for (hospital) in m.locations if hospital.type == CSC]
        push!(lst, "STAY")
        return lst
    
    # If patient has already been routed to a CSC, can't be transferred anywhere else
    elseif s.loc.type == CSC   
        return ["STAY"]
    end
end

# Using PyCall here: these are two python functions
# py"""
# import requests

# def parse_result(duration_text):
#   parts = duration_text.split()  # Split by spaces

#   n_hours = 0
#   n_mins = 0

#   # Go through the split up te
#   for i, part in enumerate(parts):
#     if "hour" in part:
#       n_hours = int(parts[i - 1])
#     if "min" in part:
#       n_mins = int(parts[i - 1])

#   return n_hours * 60 + n_mins


# def time_between_coordinates(origin, destination, API_KEY):
#     # Base URL for the Distance Matrix API endpoint
#     BASE_URL = "https://maps.googleapis.com/maps/api/distancematrix/json"
    
#     # params for request
#     params = {
#         "origins": f"{origin[0]},{origin[1]}",     # Convert the origin tuple to "lat,lng" format
#         "destinations": f"{destination[0]},{destination[1]}", # Convert the destination tuple to "lat,lng" format
#         "mode": "driving",  # We want driving directions
#         "key": API_KEY     
#     }
    
#     # Make api request
#     response = requests.get(BASE_URL, params=params)
    
#     # make sure it was successful!!!
#     if response.status_code == 200:
#         data = response.json()
        
#         # get the actual time it takes (this is as a string though)
#         duration_text = data['rows'][0]['elements'][0]['duration']['text']
#         return parse_result(duration_text)  # make sure we are returning a minute value as an integer
#     else:
#         return f"Error: {response.status_code}"


# """

function calculate_travel_time_ors(origin::Tuple{Float64, Float64}, destination::Tuple{Float64, Float64})
    url = "http://localhost:8080/ors/v2/directions/driving-car"
    params = Dict(
        "start" => join(reverse(origin), ","),
        "end" => join(reverse(destination), ",")
    )
    response = HTTP.get(url, query=params)
    if response.status == 200
        data = JSON.parse(String(response.body))
        # Extract travel time (in seconds) from the response
        travel_time_seconds = data["routes"][1]["summary"]["duration"]
        return travel_time_seconds
    else
        @error "Failed to get data from OpenRouteService API"
        return nothing
    end
end

# Given an instance of our StrokeMDP and a current location, return the nearest CSC (by car travel time).
function find_nearest_CSC(m::StrokeMDP, cur_loc::Location)
    # Vector of all CSCs
    CSCs = [loc for loc in m.locations if loc.type == CSC]

    dict = Dict()
    for CSC in CSCs
        #!dist = py"time_between_coordinates"(cur_loc.latlon, CSC.latlon, m.API_KEY)
        dist = calculate_travel_time_ors(CSC, cur_loc)
        dict[dist] = CSC
    end

    min_time = minimum(keys(dict)) 
    return dict[min_time]
end

# Given an instance of our StrokeMDP and a current location, return the nearest PSC or CSC (by car travel time).
function find_nearest_PSC_or_CSC(m::StrokeMDP, cur_loc::Location)
    # Vector of all CSCs
    potentials = [loc for loc in m.locations if loc.type == CSC || loc.type == PSC]

    dict = Dict()
    for potential in potentials
        #! dist = py"time_between_coordinates"(cur_loc.latlon, potential.latlon, m.API_KEY)
        dist = calculate_travel_time_ors(potential, cur_loc)
        dict[dist] = potential
    end

    min_time = minimum(keys(dict)) 
    return dict[min_time]

end

# Given an instance of our StrokeMDP and a current location, return the nearest hospital (by car travel time).
function find_nearest_hospital(m::StrokeMDP, cur_loc::Location)
    # Vector of all CSCs
    potentials = [loc for loc in m.locations if loc.type == CSC || loc.type == PSC]

    dict = Dict()
    for potential in potentials
        #!dist = py"time_between_coordinates"(cur_loc.latlon, potential.latlon, m.API_KEY)
        dist = calculate_travel_time_ors(potential, cur_loc)
        dict[dist] = potential
    end

    min_time = minimum(keys(dict)) 
    return dict[min_time]

end

# Transition function!
function POMDPs.transition(m::StrokeMDP, s::PatientState, a::Action)
    cur_loc = s.loc
    if a == STAY
        dest_loc = s.loc
    else
        full_term = enum_to_string(a)  # Convert action to string form
        dest_loc_name = replace(full_term, "ROUTE_" => "")

        # Search for hospital by name field in m.locations
        index = findfirst(loc -> loc.name == dest_loc_name, m.locations)
        if index === nothing
            error("Destination location not found: $dest_loc_name")
        end
        dest_loc = m.locations[index]
    end

    treatment_time = dest_loc.performance_metric  # Assuming treatment happens at destination

    #!travel_time = py"time_between_coordinates"(cur_loc.latlon, dest_loc.latlon, m.API_KEY)
    travel_time = calculate_travel_time_ors(cur_loc, dest_loc)

    t_onset = s.t_onset + treatment_time + travel_time
    
    known = s.stroke_type_known
    if a != STAY
        known = KNOWN
    end

    next_state = PatientState(dest_loc, t_onset, known, s.stroke_type)
    return Deterministic(next_state) 
end

    
# Reward function
# CITATION: Holodinsky JK, Williamson TS, Demchuk AM, et al. Modeling Stroke Patient
# Transport for All Patients With Suspected Large-Vessel Occlusion.
function POMDPs.reward(m::StrokeMDP, s::PatientState, a::Action, sp::PatientState)
    # what hospital type are we going to?
    if sp.loc.type == CSC
        t_onset_puncture = sp.t_onset
        t_onset_needle = sp.t_onset
    elseif sp.loc.type == PSC
        # find nearest CSC; calculate time to CSC
        nearest_CSC = find_nearest_CSC(m, sp.loc)
        #!time_to_CSC = py"time_between_coordinates"(sp.loc.latlon, nearest_CSC.latlon, m.API_KEY)
        time_to_CSC = calculate_travel_time_ors(sp.loc, nearest_CSC)
        t_onset_puncture = sp.t_onset + time_to_CSC + nearest_CSC.performance_metric
        t_onset_needle = sp.t_onset
    elseif sp.loc.type == CLINIC || sp.loc.type == FIELD
        # find nearest CSC; calculate time to CSC
        nearest_CSC = find_nearest_CSC(m, sp.loc)
        #!time_to_CSC = py"time_between_coordinates"(sp.loc.latlon, nearest_CSC.latlon, m.API_KEY)
        time_to_CSC = calculate_travel_time_ors(sp.loc, nearest_CSC) 
        t_onset_puncture = sp.t_onset + time_to_CSC + nearest_CSC.performance_metric

        # find nearest CSC or PSC; calculate time to CSC/PSC
        nearest_PSC_or_CSC = find_nearest_PSC_or_CSC(m, sp.loc)
        #!time_to_PSC_or_CSC = py"time_between_coordinates"(sp.loc.latlon, nearest_PSC_or_CSC.latlon, m.API_KEY)
        time_to_PSC_or_CSC = calculate_travel_time_ors(sp.loc, nearest_PSC_or_CSC)
        t_onset_needle = sp.t_onset + time_to_PSC_or_CSC + nearest_PSC_or_CSC.performance_metric
    end
        
    if s.stroke_type_known == KNOWN
        if s.stroke_type == LVO
            if t_onset_needle < 270
                prob_altepase = 0.2359 + 0.0000002(t_onset_needle)^2 - 0.0004(t_onset_needle) 
            else
                prob_altepase = 0.1328 
            end
        
            if t_onset_puncture < 270
                prob_EVT = 0.3394 + 0.00000004(t_onset_puncture)^2 - 0.0002(t_onset_puncture) 
            else
                prob_EVT = 0.129 
            end
            p_good_outcome = prob_altepase + ((1 - prob_altepase) * prob_EVT)
        elseif s.stroke_type == NLVO
            if t_onset_needle < 270
                p_good_outcome = 0.6343 - 0.00000005(t_onset_needle)^2 - 0.0005(t_onset_needle)
            else
                p_good_outcome = 0.4622 
            end
        elseif s.stroke_type == HEMORRHAGIC
            p_good_outcome = 0.24
        elseif s.stroke_type == MIMIC
            p_good_outcome = 0.90
        end
    else
        # calculate p_good_outcome_LVO
        if t_onset_needle < 270
            prob_altepase = 0.2359 + 0.0000002(t_onset_needle)^2 - 0.0004(t_onset_needle) 
        else
            prob_altepase = 0.1328 
        end

        if t_onset_puncture < 270
            prob_EVT = 0.3394 + 0.00000004(t_onset_puncture)^2 - 0.0002(t_onset_puncture) 
        else
            prob_EVT = 0.129 
        end

        p_good_outcome_LVO = prob_altepase + ((1 - prob_altepase) * prob_EVT)


        # calculate p_good_outcome_nLVO
        if t_onset_needle < 270
            p_good_outcome_nLVO = 0.6343 - 0.00000005(t_onset_needle)^2 - 0.0005(t_onset_needle)
        else
            p_good_outcome_nLVO = 0.4622 
        end


        # calculate p_good_outcome_hemorragic
        p_good_outcome_hemhorragic  = 0.24

        # calculate p_good_outcome_mimic
        p_good_outcome_mimic  = 0.90

        p_good_outcome = m.p_LVO*p_good_outcome_LVO + m.p_nLVO*p_good_outcome_nLVO + m.p_Hemorrhagic*p_good_outcome_hemhorragic 
        + m.p_Mimic*p_good_outcome_mimic
    end
    return p_good_outcome
end


function forward_search(m::StrokeMDP, s::PatientState, depth::Int)
    if depth == 0
        return 0.0
    end
    
    best_value = -Inf
    for a in actions(m, s)
        a = string_to_enum(a)
        sp_wrapper = transition(m, s, a) # assuming deterministic
        sp = rand(sp_wrapper) # Sample the deterministic state
        r = reward(m, s, a, sp)
        value = r + discount(m) * forward_search(m, sp, depth-1)
        best_value = max(best_value, value)
    end
    
    return best_value
end

function best_action(m::StrokeMDP, s::PatientState, depth::Int)
    best_act = nothing
    best_value = -Inf
    
    for a_str in actions(m, s)
        #println("***** a_str is:", a_str)
        a = string_to_enum(a_str) # Convert string to Action type
        sp_wrapper = transition(m, s, a) # assuming deterministic
        sp = rand(sp_wrapper) # Sample the deterministic state
        r = reward(m, s, a, sp)
        value = r + discount(m) * forward_search(m, sp, depth-1)
        if value > best_value
            best_value = value
            best_act = a
        end
    end
    
    return best_act
end


function current_CApolicy_action(m::StrokeMDP, s::PatientState)
    cur_loc = s.loc
    nearest_hospital = find_nearest_hospital(m, cur_loc)
    return "ROUTE_" * nearest_hospital.name
end

