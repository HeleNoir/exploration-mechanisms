@echo off
::##############################################################################
:: BAT version of target-runner for Windows.
:: Contributed by Andre de Souza Andrade <andre.andrade@uniriotec.br>.
:: Check other examples in examples/
::
:: This script is run in the execution directory (execDir, --exec-dir).
::
:: PARAMETERS:
:: %%1 is the candidate configuration number
:: %%2 is the instance ID
:: %%3 is the seed
:: %%4 is the instance name
:: The rest are parameters to the target-algorithm
::
:: RETURN VALUE:
:: This script should print one numerical value: the cost that must be minimized.
:: Exit with 0 if no error, with 1 in case of error
::##############################################################################

:: Please change the EXE and FIXED_PARAMS to the correct ones
SET "exe=cargo run --release --package exploration-mechanisms --bin irace_random_restart_pso --"
:: TODO update PSO parameters
SET "fixed_params= --population-size 50 --inertia-weight 0.7 --c1 1.2 --c2 1.2"

FOR /f "tokens=1-4*" %%a IN ("%*") DO (
	SET candidate=%%a
	SET instance_id=%%b
	SET seed=%%c
	SET instance=%%d
	SET candidate_parameters=%%e
)

SET "stdout=%candidate%-%instance_id%-%seed%.stdout"
SET "stderr=%candidate%-%instance_id%-%seed%.stderr"

:: Save the output to a file, and parse the result from it.

%exe% --seed %seed% --inst %instance% %fixed_params% %candidate_parameters% 1> %stdout% 2> %stderr%


:: Reading a number from the output.
:: It assumes that the objective value is the first number in
:: the first column of the third to last line of the output and
:: the time is the first number in the first column of the second to last line.
setlocal EnableDelayedExpansion
:: Initialize variables
set "last2="
set "last1="
set "current="

:: Read the file line by line
for /f "tokens=*" %%A in (%stdout%) do (
    set "last2=!last1!"
    set "last1=!current!"
    set "current=%%A"
)

:: Now last2 contains the second-to-last line and last1 contains the third-to-last
set "COST=!last2!"
set "TIME=!last1!"

echo %COST% %TIME%

:: Un-comment this if you want to delete temporary files.
del %stdout% %stderr%
exit 0
