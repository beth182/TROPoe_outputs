#!/bin/sh

# $Id: run_tropoe_ops.csh,v 1.2 2022/03/23 22:20:01 dave.turner Exp $

# This script makes mounting/running the TropOE container "easy" for operations

if [[ $# -ne 9 ]]; then 
  echo "USAGE: $0 yyyymmdd vip_file prior_file shour ehour verbose data_path temporary_path image_id"
  echo "   where      yyyymmdd : is the date to process"
  echo "              vip_file : is the path/name of the VIP file"
  echo "            prior_file : is the path/name of the prior data file"
  echo "                 shour : is the start hour"
  echo "                 ehour : is the  end  hour"
  echo "               verbose : is the verbosity level (1, 2, or 3)"
  echo "             data_path : is the location of the data tree"
  echo "        temporary_path : is the location of the (fast) temporary directory used for the LBLRTM and MonoRTM runs"
  echo "   and        image_id : is the identification number or name of the Docker image to execute"
  exit
fi

echo "Running TROPoe in operational mode, with"
echo "                            Date (yyyymmdd) : $1"
echo "                                   VIP file : $2"
echo "                                 Prior file : $3"
echo "                                 Start hour : $4"
echo "                                   End hour : $5"
echo "                               Verbose flag : $6"
echo "  External data directory (mapped to /data) : $7"
echo "  External temp directory (mapped to /tmp2) : $8"
echo "                                 Image name : $9"


# If using Docker, then use this command
echo "Running image Docker"
docker run -it --userns=host -e "app=TROPoe" -e "yyyymmdd=$1" -e "vfile=$2" -e "pfile=$3" -e "shour=$4" -e "ehour=$5" -e "verbose=$6" -v $7:/data -v $8:/tmp2 $9 

# Else if you are using Podman, then comment the above command and uncomment this one
#echo "Running image Podman"
#podman run -it -u root --rm -e "app=TROPoe" -e "yyyymmdd=$1" -e "vfile=$2" -e "pfile=$3" -e "shour=$4" -e "ehour=$5" -e "verbose=$6" --security-opt label=disable -v $7:/data -v $8:/tmp2 $9 
