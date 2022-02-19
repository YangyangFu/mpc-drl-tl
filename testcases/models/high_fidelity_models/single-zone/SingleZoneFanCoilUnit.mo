within ;
package SingleZoneFanCoilUnit
  "For BOPTEST, the simple single zone testcase based on BESTEST with an air-based HVAC system."
  package TestCases "Package to hold test case models"
  extends Modelica.Icons.ExamplesPackage;
    model FanControl
      "Testcase model with ideal airflow controlled by external fan control signal"
      extends Modelica.Icons.Example;
      BaseClasses.Case900FF zon(mAir_flow_nominal=fcu.mAir_flow_nominal)
        annotation (Placement(transformation(extent={{34,-10},{54,10}})));

      BaseClasses.FanCoilUnit_T fcu "Fan coil unit"
        annotation (Placement(transformation(extent={{-20,-8},{0,20}})));
      Modelica.Blocks.Sources.Constant TSupAirSet(k=273.15 + 14)
        annotation (Placement(transformation(extent={{-80,30},{-60,50}})));
      Modelica.Blocks.Interfaces.RealInput uFan "Fan speed signal"
        annotation (Placement(transformation(extent={{-140,-20},{-100,20}})));
    equation
      connect(fcu.supplyAir, zon.supplyAir) annotation (Line(points={{0,13.7778},
              {20,13.7778},{20,2},{34,2}},color={0,127,255}));
      connect(fcu.returnAir, zon.returnAir) annotation (Line(points={{0,-6.44444},{
              20,-6.44444},{20,-2},{34,-2}}, color={0,127,255}));
      connect(TSupAirSet.y, fcu.TSup) annotation (Line(points={{-59,40},{-40,40},
              {-40,9.11111},{-21.4286,9.11111}}, color={0,0,127}));
      connect(fcu.uFan, uFan) annotation (Line(points={{-21.4286,2.88889},{-120,
              2.88889},{-120,0}}, color={0,0,127}));
      annotation (Icon(coordinateSystem(preserveAspectRatio=false)), Diagram(
            coordinateSystem(preserveAspectRatio=false)),
        experiment(
          StopTime=31536000,
          Interval=300,
          Tolerance=1e-06,
          __Dymola_Algorithm="Cvode"),
    Documentation(info="<html>
General model description.
<h3>Building Design and Use</h3>
<h4>Architecture</h4>
<p>
The building is a single room based on the BESTEST Case 900 model definition.
The floor dimensions are 6m x 8m and the floor-to-ceiling height is 2.7m.
There are four exterior walls facing the cardinal directions and a flat roof.
The walls facing east-west have the short dimension.  The south wall contains
two windows, each 3m wide and 2m tall.  The use of the building is assumed
to be a two-person office with a light load density.
</p>
<h4>Constructions</h4>
<p>
The constructions are based on the BESTEST Case 900 model definition.  The
exterior walls are made of concrete block and insulation, while the floor
is a concrete slab.  The roof is made of wood frame with insulation.  The
layer-by-layer specifications are (Outside to Inside):
</p>
<p>
<b>Exterior Walls</b>
<table>
  <tr>
  <th>Name</th>
  <th>Thickness [m]</th>
  <th>Thermal Conductivity [W/m-K]</th>
  <th>Specific Heat Capacity [J/kg-K]</th>
  <th>Density [kg/m3]</th>
  </tr>
  <tr>
  <td>Layer 1</td>
  <td>0.009</td>
  <td>0.140</td>
  <td>900</td>
  <td>530</td>
  </tr>
  <tr>
  <td>Layer 2</td>
  <td>0.0615</td>
  <td>0.040</td>
  <td>1400</td>
  <td>10</td>
  </tr>
  <tr>
  <td>Layer 3</td>
  <td>0.100</td>
  <td>0.510</td>
  <td>1000</td>
  <td>1400</td>
  </tr>
  </table>
<table>
  <tr>
  <th>Name</th>
  <th>IR Emissivity [-]</th>
  <th>Solar Emissivity [-]</th>
  </tr>
  <tr>
  <td>Outside</td>
  <td>0.9</td>
  <td>0.6</td>
  </tr>
  <tr>
  <td>Inside</td>
  <td>0.9</td>
  <td>0.6</td>
  </tr>
</table>
</p>
<p>
<b>Floor</b>
<table>
  <tr>
  <th>Name</th>
  <th>Thickness [m]</th>
  <th>Thermal Conductivity [W/m-K]</th>
  <th>Specific Heat Capacity [J/kg-K]</th>
  <th>Density [kg/m3]</th>
  </tr>
  <tr>
  <td>Layer 1</td>
  <td>1.007</td>
  <td>0.040</td>
  <td>0</td>
  <td>0</td>
  </tr>
  <tr>
  <td>Layer 2</td>
  <td>0.080</td>
  <td>1.130</td>
  <td>1000</td>
  <td>1400</td>
  </tr>
</table>
<table>
  <tr>
  <th>Name</th>
  <th>IR Emissivity [-]</th>
  <th>Solar Emissivity [-]</th>
  </tr>
  <tr>
  <td>Outside</td>
  <td>0.9</td>
  <td>0.6</td>
  </tr>
  <tr>
  <td>Inside</td>
  <td>0.9</td>
  <td>0.6</td>
  </tr>
</table>
</p>
<p>
<b>Roof</b>
<table>
  <tr>
  <th>Name</th>
  <th>Thickness [m]</th>
  <th>Thermal Conductivity [W/m-K]</th>
  <th>Specific Heat Capacity [J/kg-K]</th>
  <th>Density [kg/m3]</th>
  </tr>
  <tr>
  <td>Layer 1</td>
  <td>0.019</td>
  <td>0.140</td>
  <td>900</td>
  <td>530</td>
  </tr>
  <tr>
  <td>Layer 2</td>
  <td>0.1118</td>
  <td>0.040</td>
  <td>840</td>
  <td>12</td>
  </tr>
  <tr>
  <td>Layer 3</td>
  <td>0.010</td>
  <td>0.160</td>
  <td>840</td>
  <td>950</td>
  </tr>
</table>
<table>
  <tr>
  <th>Name</th>
  <th>IR Emissivity [-]</th>
  <th>Solar Emissivity [-]</th>
  </tr>
  <tr>
  <td>Outside</td>
  <td>0.9</td>
  <td>0.6</td>
  </tr>
  <tr>
  <td>Inside</td>
  <td>0.9</td>
  <td>0.6</td>
  </tr>
</table>
<p>
The windows are double pane clear 3.175mm glass with 13mm air gap.
</p>

<h4>Occupancy schedules</h4>
<p>
There is maximum occupancy (two people) from 8am to 6pm each day,
and no occupancy during all other times.
</p>
<h4>Internal loads and schedules</h4>
<p>
The internal heat gains from plug loads come mainly from computers and monitors.
The internal heat gains from lighting come from hanging fluorescent fixtures.
Both types of loads are at maximum during occupied periods and 0.1 maximum
during all other times.  The occupied heating and cooling temperature
setpoints are 21 C and 24 C respectively, while the unoccupied heating
and cooling temperature setpoints are 15 C and 30 C respectively.

</p>
<h4>Climate data</h4>
<p>
The climate is assumed to be near Denver, CO, USA with a latitude and
longitude of 39.76,-104.86.  The climate data comes from the
Denver-Stapleton,CO,USA,TMY.
</p>
<h3>HVAC System Design</h3>
<h4>Primary and secondary system designs</h4>
<p>
Heating and cooling is provided to the office using an idealized four-pipe
fan coil unit (FCU), presented in Figure 1 below.
The FCU contains a fan, cooling coil, heating coil,
and filter.  The fan draws room air into the unit, blows it over the coils
and through the filter, and supplies the conditioned air back to the room.
There is a variable speed drive serving the fan motor.  The cooling coil
is served by chilled water produced by a chiller and the heating coil is
served by hot water produced by a gas boiler.
</p>

<p>
<br>
</p>

<p>
<img src=\"../../../doc/images/Schematic.png\"/>
<figcaption><small>Figure 1: System schematic.</small></figcaption>
</p>

<p>
<br>
</p>

<h4>Equipment specifications and performance maps</h4>
<p>
For the fan, the design airflow rate is 0.55 kg/s and design pressure rise is
185 Pa.  The fan and motor efficiencies are both constant at 0.7.
The heat from the motor is added to the air stream.

The COP of the chiller is assumed constant at 3.0.  The efficiency of the
gas boiler is assumed constant at 0.9.
</p>
<h4>Rule-based or local-loop controllers (if included)</h4>
<p>
A baseline thermostat controller provides heating and cooling as necessary
to the room by modulating the supply air temperature and
fan speed.  The thermostat, designated as C1 in Figure 1 and shown in Figure 2 below,
uses two different PI controllers for heating and
cooling, each taking the respective zone temperature set point and zone
temperature measurement as inputs.  The outputs are used to control supply air
temperature set point and fan speed according to the map shown in Figure 3 below.
The supply air temperature is exactly met by the coils using an ideal controller
depicted as C2 in Figure 1.
For heating, the maximum supply air temperature is 40 C and the minimum is the
zone occupied heating temperature setpoint.  For cooling, the minimum supply
air temperature is 12 C and the maximum is the zone occupied cooling
temperature setpoint.
</p>

<p>
<br>
</p>

<p>
<img src=\"../../../doc/images/C1.png\"/>
<figcaption><small>Figure 2: Controller C1.</small></figcaption>
</p>

<p>
<br>
</p>

<p>
<img src=\"../../../doc/images/ControlSchematic_Ideal.png\" width=600 />
<figcaption><small>Figure 3: Mapping of PI output to supply air temperature set point and fan speed in controller C1.</small></figcaption>
</p>

<p>
<br>
</p>

<h3>Model IO's</h3>
<h4>Inputs</h4>
The model inputs are:
<ul>
<li>
<code>fcu_oveTSup_u</code> [K] [min=285.15, max=313.15]: Supply air temperature setpoint
</li>
<li>
<code>fcu_oveFan_u</code> [1] [min=0.0, max=1.0]: Fan control signal as air mass flow rate normalized to the design air mass flow rate
</li>
<li>
<code>con_oveTSetHea_u</code> [K] [min=288.15, max=296.15]: Zone temperature setpoint for heating
</li>
<li>
<code>con_oveTSetCoo_u</code> [K] [min=296.15, max=303.15]: Zone temperature setpoint for cooling
</li>
</ul>
<h4>Outputs</h4>
The model outputs are:
<ul>
<li>
<code>fcu_reaFloSup_y</code> [kg/s] [min=None, max=None]: Supply air mass flow rate
</li>
<li>
<code>fcu_reaPCoo_y</code> [W] [min=None, max=None]: Cooling electrical power consumption
</li>
<li>
<code>fcu_reaPFan_y</code> [W] [min=None, max=None]: Supply fan electrical power consumption
</li>
<li>
<code>fcu_reaPHea_y</code> [W] [min=None, max=None]: Heating thermal power consumption
</li>
<li>
<code>zon_reaCO2RooAir_y</code> [ppm] [min=None, max=None]: Zone air CO2 concentration
</li>
<li>
<code>zon_reaPLig_y</code> [W] [min=None, max=None]: Lighting power submeter
</li>
<li>
<code>zon_reaPPlu_y</code> [W] [min=None, max=None]: Plug load power submeter
</li>
<li>
<code>zon_reaTRooAir_y</code> [K] [min=None, max=None]: Zone air temperature
</li>
<li>
<code>zon_weaSta_reaWeaCeiHei_y</code> [m] [min=None, max=None]: Cloud cover ceiling height measurement
</li>
<li>
<code>zon_weaSta_reaWeaCloTim_y</code> [s] [min=None, max=None]: Day number with units of seconds
</li>
<li>
<code>zon_weaSta_reaWeaHDifHor_y</code> [W/m2] [min=None, max=None]: Horizontal diffuse solar radiation measurement
</li>
<li>
<code>zon_weaSta_reaWeaHDirNor_y</code> [W/m2] [min=None, max=None]: Direct normal radiation measurement
</li>
<li>
<code>zon_weaSta_reaWeaHGloHor_y</code> [W/m2] [min=None, max=None]: Global horizontal solar irradiation measurement
</li>
<li>
<code>zon_weaSta_reaWeaHHorIR_y</code> [W/m2] [min=None, max=None]: Horizontal infrared irradiation measurement
</li>
<li>
<code>zon_weaSta_reaWeaLat_y</code> [rad] [min=None, max=None]: Latitude of the location
</li>
<li>
<code>zon_weaSta_reaWeaLon_y</code> [rad] [min=None, max=None]: Longitude of the location
</li>
<li>
<code>zon_weaSta_reaWeaNOpa_y</code> [1] [min=None, max=None]: Opaque sky cover measurement
</li>
<li>
<code>zon_weaSta_reaWeaNTot_y</code> [1] [min=None, max=None]: Sky cover measurement
</li>
<li>
<code>zon_weaSta_reaWeaPAtm_y</code> [Pa] [min=None, max=None]: Atmospheric pressure measurement
</li>
<li>
<code>zon_weaSta_reaWeaRelHum_y</code> [1] [min=None, max=None]: Outside relative humidity measurement
</li>
<li>
<code>zon_weaSta_reaWeaSolAlt_y</code> [rad] [min=None, max=None]: Solar altitude angle measurement
</li>
<li>
<code>zon_weaSta_reaWeaSolDec_y</code> [rad] [min=None, max=None]: Solar declination angle measurement
</li>
<li>
<code>zon_weaSta_reaWeaSolHouAng_y</code> [rad] [min=None, max=None]: Solar hour angle measurement
</li>
<li>
<code>zon_weaSta_reaWeaSolTim_y</code> [s] [min=None, max=None]: Solar time
</li>
<li>
<code>zon_weaSta_reaWeaSolZen_y</code> [rad] [min=None, max=None]: Solar zenith angle measurement
</li>
<li>
<code>zon_weaSta_reaWeaTBlaSky_y</code> [K] [min=None, max=None]: Black-body sky temperature measurement
</li>
<li>
<code>zon_weaSta_reaWeaTDewPoi_y</code> [K] [min=None, max=None]: Dew point temperature measurement
</li>
<li>
<code>zon_weaSta_reaWeaTDryBul_y</code> [K] [min=None, max=None]: Outside drybulb temperature measurement
</li>
<li>
<code>zon_weaSta_reaWeaTWetBul_y</code> [K] [min=None, max=None]: Wet bulb temperature measurement
</li>
<li>
<code>zon_weaSta_reaWeaWinDir_y</code> [rad] [min=None, max=None]: Wind direction measurement
</ul>
<h3>Additional System Design</h3>
<h4>Lighting</h4>
<p>
Artificial lighting is provided by hanging fluorescent fixtures.
</p>
<h4>Shading</h4>
<p>
There are no shades on the building.
</p>
<h4>Onsite Generation and Storage</h4>
<p>
There is no energy generation or storage on the site.
</p>
<h3>Model Implementation Details</h3>
<h4>Moist vs. dry air</h4>
<p>
A moist air model is used, but condensation is not modeled on the cooling coil
and humidity is not monitored.

</p>
<h4>Pressure-flow models</h4>
<p>
The FCU fan is speed-controlled and the resulting flow is calculated based
on resulting pressure rise by the fan and fixed pressure drop of the system.
</p>
<h4>Infiltration models</h4>
<p>
A constant infiltration flowrate is assumed to be 0.5 ACH.
</p>
<h4>Other assumptions</h4>
<p>
The supply air temperature is directly specified.
</p>
<p>
CO2 generation is 0.0048 L/s per person (Table 5, Persily and De Jonge 2017)
and density of CO2 assumed to be 1.8 kg/m^3,
making CO2 generation 8.64e-6 kg/s per person.
Outside air CO2 concentration is 400 ppm.  However, CO2 concentration
is not controlled for in the model.
</p>
<p>
Persily, A. and De Jonge, L. (2017).
Carbon dioxide generation rates for building occupants.
Indoor Air, 27, 868–879.  https://doi.org/10.1111/ina.12383.
</p>
<h3>Scenario Information</h3>
<h4>Time Periods</h4>
<p>
The <b>Peak Heat Day</b> (specifier for <code>/scenario</code> API is <code>'peak_heat_day'</code>) period is:
<ul>
This testing time period is a two-week test with one-week warmup period utilizing
baseline control.  The two-week period is centered on the day with the
maximum 15-minute system heating load in the year.
</ul>
<ul>
Start Time: Day 334.
</ul>
<ul>
End Time: Day 348.
</ul>
</p>
<p>
The <b>Typical Heat Day</b> (specifier for <code>/scenario</code> API is <code>'typical_heat_day'</code>) period is:
<ul>
This testing time period is a two-week test with one-week warmup period utilizing
baseline control.  The two-week period is centered on the day with day with
the maximum 15-minute system heating load that is closest from below to the
median of all 15-minute maximum heating loads of all days in the year.
</ul>
<ul>
Start Time: Day 44.
</ul>
<ul>
End Time: Day 58.
</ul>
</p>
<p>
The <b>Peak Cool Day</b> (specifier for <code>/scenario</code> API is <code>'peak_cool_day'</code>) period is:
<ul>
This testing time period is a two-week test with one-week warmup period utilizing
baseline control.  The two-week period is centered on the day with the
maximum 15-minute system cooling load in the year.
</ul>
<ul>
Start Time: Day 282.
</ul>
<ul>
End Time: Day 296.
</ul>
</p>
<p>
The <b>Typical Cool Day</b> (specifier for <code>/scenario</code> API is <code>'typical_cool_day'</code>) period is:
<ul>
This testing time period is a two-week test with one-week warmup period utilizing
baseline control.  The two-week period is centered on the day with day with
the maximum 15-minute system cooling load that is closest from below to the
median of all 15-minute maximum cooling loads of all days in the year.
</ul>
<ul>
Start Time: Day 146.
</ul>
<ul>
End Time: Day 160.
</ul>
</p>
<p>
The <b>Mix Day</b> (specifier for <code>/scenario</code> API is <code>'mix_day'</code>) period is:
<ul>
This testing time period is a two-week test with one-week warmup period utilizing
baseline control.  The two-week period is centered on the day with the maximimum
sum of daily heating and cooling loads minus the difference between
daily heating and cooling loads.  This is a day with both significant heating
and cooling loads.
</ul>
<ul>
Start Time: Day 14.
</ul>
<ul>
End Time: Day 28.
</ul>
</p>
<h4>Energy Pricing</h4>
<p>
The <b>Constant Electricity Price</b> (specifier for <code>/scenario</code> API is <code>'constant'</code>) profile is:
<ul>
Based on the Schedule R tariff
for winter season and summer season first 500 kWh as defined by the
utility servicing the assumed location of the test case.  It is $0.05461/kWh.
For reference,
see https://www.xcelenergy.com/company/rates_and_regulations/rates/rate_books
in the section on Current Tariffs/Electric Rate Books (PDF).
</ul>
<ul>
Specifier for <code>/scenario</code> API is <code>'constant'</code>.
</ul>
</p>
<p>
The <b>Dynamic Electricity Price</b> (specifier for <code>/scenario</code> API is <code>'dynamic'</code>) profile is:
<ul>
Based on the Schedule RE-TOU tariff
as defined by the utility servicing the assumed location of the test case.
For reference,
see https://www.xcelenergy.com/company/rates_and_regulations/rates/rate_books
in the section on Current Tariffs/Electric Rate Books (PDF).
</ul>
</p>
<p>
<ul>
<li>
Summer on-peak is $0.13814/kWh.
</li>
<li>
Summer mid-peak is $0.08420/kWh.
</li>
<li>
Summer off-peak is $0.04440/kWh.
</li>
<li>
Winter on-peak is $0.08880/kWh.
</li>
<li>
Winter mid-peak is $0.05413/kWh.
</li>
<li>
Winter off-peak is $0.04440/kWh.
</li>
<li>
The Summer season is June 1 to September 30.
</li>
<li>
The Winter season is October 1 to May 31.
</li>
</p>
<p>
<u>The On-Peak Period is</u>:
<ul>
<li>
Summer and Winter weekdays except Holidays, between 2:00 p.m. and 6:00 p.m.
local time.
</li>
</ul>
<u>The Mid-Peak Period is</u>:
<ul>
<li>
Summer and Winter weekdays except Holidays, between 9:00 a.m. and
2:00 p.m. and between 6:00 p.m. and 9:00 p.m. local time.
</li>
<li>
Summer and Winter weekends and Holidays, between 9:00 a.m. and
9:00 p.m. local time.
</li>
</ul>
<u>The Off-Peak Period is</u>:
<ul>
<li>
Summer and Winter daily, between 9:00 p.m. and 9:00 a.m. local time.
</li>
</ul>
</ul>
</p>
<p>
The <b>Highly Dynamic Electricity Price</b> (specifier for <code>/scenario</code> API is <code>'highly_dynamic'</code>) profile is:
<ul>
Based on the the
day-ahead energy prices (LMP) as determined in the Southwest Power Pool
wholesale electricity market for node LAM345 in the year 2018.
For reference,
see https://marketplace.spp.org/pages/da-lmp-by-location#%2F2018.
</ul>
</p>
<p>
The <b>Gas Price</b> profile is:
<ul>
Based on the Schedule R tariff for usage price per therm as defined by the
utility servicing the assumed location of the test case.  It is $0.002878/kWh
($0.0844/therm).
For reference,
see https://www.xcelenergy.com/company/rates_and_regulations/rates/rate_books
in the section on Summary of Gas Rates for 10/1/19.
</ul>
</p>
<h4>Emission Factors</h4>
<p>
The <b>Electricity Emissions Factor</b> profile is:
<ul>
Based on the average electricity generation mix for CO,USA for the year of
2017.  It is 0.6618 kgCO2/kWh (1459 lbsCO2/MWh).
For reference,
see https://www.eia.gov/electricity/state/colorado/.
</ul>
</p>
<p>
The <b>Gas Emissions Factor</b> profile is:
<ul>
Based on the kgCO2 emitted per amount of natural gas burned in terms of
energy content.  It is 0.18108 kgCO2/kWh (53.07 kgCO2/milBTU).
For reference,
see https://www.eia.gov/environment/emissions/co2_vol_mass.php.
</ul>
</p>
</html>",
    revisions="<html>
<ul>
<li>
December 6, 2021, by David Blum:<br/>
Correct mix day time period.
This is for <a href=https://github.com/ibpsa/project1-boptest/issues/381>
BOPTEST issue #381</a>.
</li>
<li>
April 13, 2021, by David Blum:<br/>
Add time period documentation.
</li>
<li>
November 10, 2020, by David Blum:<br/>
Add weather station measurements.
</li>
<li>
March 4, 2020, by David Blum:<br/>
Updated CO2 generation per person and method of ppm calculation.
</li>
<li>
December 15, 2019, by David Blum:<br/>
First implementation.
</li>
</ul>
</html>"));
    end FanControl;

    model Baseline "Testcase model with ideal airflow"
      extends Modelica.Icons.Example;
      BaseClasses.Case900FF zon(mAir_flow_nominal=fcu.mAir_flow_nominal)
        annotation (Placement(transformation(extent={{34,-10},{54,10}})));

      BaseClasses.Thermostat_T con "Thermostat controller"
        annotation (Placement(transformation(extent={{-80,-10},{-60,10}})));
      BaseClasses.FanCoilUnit_T fcu "Fan coil unit"
        annotation (Placement(transformation(extent={{-20,-8},{0,20}})));
    equation
      connect(fcu.supplyAir, zon.supplyAir) annotation (Line(points={{0,13.7778},
              {20,13.7778},{20,2},{34,2}},color={0,127,255}));
      connect(fcu.returnAir, zon.returnAir) annotation (Line(points={{0,-6.44444},{
              20,-6.44444},{20,-2},{34,-2}}, color={0,127,255}));
      connect(zon.TRooAir, con.TZon) annotation (Line(points={{61,0},{80,0},{80,-40},
              {-100,-40},{-100,0},{-82,0}}, color={0,0,127}));
      connect(con.TSup, fcu.TSup) annotation (Line(points={{-59,6},{-44,6},{-44,
              9.11111},{-21.4286,9.11111}}, color={0,0,127}));
      connect(con.yFan, fcu.uFan) annotation (Line(points={{-59,0},{-44,0},{-44,
              2.88889},{-21.4286,2.88889}}, color={0,0,127}));
      annotation (Icon(coordinateSystem(preserveAspectRatio=false)), Diagram(
            coordinateSystem(preserveAspectRatio=false)),
        experiment(
          StopTime=31536000,
          Interval=300,
          Tolerance=1e-06,
          __Dymola_Algorithm="Cvode"),
    Documentation(info="<html>
General model description.
<h3>Building Design and Use</h3>
<h4>Architecture</h4>
<p>
The building is a single room based on the BESTEST Case 900 model definition.
The floor dimensions are 6m x 8m and the floor-to-ceiling height is 2.7m.
There are four exterior walls facing the cardinal directions and a flat roof.
The walls facing east-west have the short dimension.  The south wall contains
two windows, each 3m wide and 2m tall.  The use of the building is assumed
to be a two-person office with a light load density.
</p>
<h4>Constructions</h4>
<p>
The constructions are based on the BESTEST Case 900 model definition.  The
exterior walls are made of concrete block and insulation, while the floor
is a concrete slab.  The roof is made of wood frame with insulation.  The
layer-by-layer specifications are (Outside to Inside):
</p>
<p>
<b>Exterior Walls</b>
<table>
  <tr>
  <th>Name</th>
  <th>Thickness [m]</th>
  <th>Thermal Conductivity [W/m-K]</th>
  <th>Specific Heat Capacity [J/kg-K]</th>
  <th>Density [kg/m3]</th>
  </tr>
  <tr>
  <td>Layer 1</td>
  <td>0.009</td>
  <td>0.140</td>
  <td>900</td>
  <td>530</td>
  </tr>
  <tr>
  <td>Layer 2</td>
  <td>0.0615</td>
  <td>0.040</td>
  <td>1400</td>
  <td>10</td>
  </tr>
  <tr>
  <td>Layer 3</td>
  <td>0.100</td>
  <td>0.510</td>
  <td>1000</td>
  <td>1400</td>
  </tr>
  </table>
<table>
  <tr>
  <th>Name</th>
  <th>IR Emissivity [-]</th>
  <th>Solar Emissivity [-]</th>
  </tr>
  <tr>
  <td>Outside</td>
  <td>0.9</td>
  <td>0.6</td>
  </tr>
  <tr>
  <td>Inside</td>
  <td>0.9</td>
  <td>0.6</td>
  </tr>
</table>
</p>
<p>
<b>Floor</b>
<table>
  <tr>
  <th>Name</th>
  <th>Thickness [m]</th>
  <th>Thermal Conductivity [W/m-K]</th>
  <th>Specific Heat Capacity [J/kg-K]</th>
  <th>Density [kg/m3]</th>
  </tr>
  <tr>
  <td>Layer 1</td>
  <td>1.007</td>
  <td>0.040</td>
  <td>0</td>
  <td>0</td>
  </tr>
  <tr>
  <td>Layer 2</td>
  <td>0.080</td>
  <td>1.130</td>
  <td>1000</td>
  <td>1400</td>
  </tr>
</table>
<table>
  <tr>
  <th>Name</th>
  <th>IR Emissivity [-]</th>
  <th>Solar Emissivity [-]</th>
  </tr>
  <tr>
  <td>Outside</td>
  <td>0.9</td>
  <td>0.6</td>
  </tr>
  <tr>
  <td>Inside</td>
  <td>0.9</td>
  <td>0.6</td>
  </tr>
</table>
</p>
<p>
<b>Roof</b>
<table>
  <tr>
  <th>Name</th>
  <th>Thickness [m]</th>
  <th>Thermal Conductivity [W/m-K]</th>
  <th>Specific Heat Capacity [J/kg-K]</th>
  <th>Density [kg/m3]</th>
  </tr>
  <tr>
  <td>Layer 1</td>
  <td>0.019</td>
  <td>0.140</td>
  <td>900</td>
  <td>530</td>
  </tr>
  <tr>
  <td>Layer 2</td>
  <td>0.1118</td>
  <td>0.040</td>
  <td>840</td>
  <td>12</td>
  </tr>
  <tr>
  <td>Layer 3</td>
  <td>0.010</td>
  <td>0.160</td>
  <td>840</td>
  <td>950</td>
  </tr>
</table>
<table>
  <tr>
  <th>Name</th>
  <th>IR Emissivity [-]</th>
  <th>Solar Emissivity [-]</th>
  </tr>
  <tr>
  <td>Outside</td>
  <td>0.9</td>
  <td>0.6</td>
  </tr>
  <tr>
  <td>Inside</td>
  <td>0.9</td>
  <td>0.6</td>
  </tr>
</table>
<p>
The windows are double pane clear 3.175mm glass with 13mm air gap.
</p>

<h4>Occupancy schedules</h4>
<p>
There is maximum occupancy (two people) from 8am to 6pm each day,
and no occupancy during all other times.
</p>
<h4>Internal loads and schedules</h4>
<p>
The internal heat gains from plug loads come mainly from computers and monitors.
The internal heat gains from lighting come from hanging fluorescent fixtures.
Both types of loads are at maximum during occupied periods and 0.1 maximum
during all other times.  The occupied heating and cooling temperature
setpoints are 21 C and 24 C respectively, while the unoccupied heating
and cooling temperature setpoints are 15 C and 30 C respectively.

</p>
<h4>Climate data</h4>
<p>
The climate is assumed to be near Denver, CO, USA with a latitude and
longitude of 39.76,-104.86.  The climate data comes from the
Denver-Stapleton,CO,USA,TMY.
</p>
<h3>HVAC System Design</h3>
<h4>Primary and secondary system designs</h4>
<p>
Heating and cooling is provided to the office using an idealized four-pipe
fan coil unit (FCU), presented in Figure 1 below.
The FCU contains a fan, cooling coil, heating coil,
and filter.  The fan draws room air into the unit, blows it over the coils
and through the filter, and supplies the conditioned air back to the room.
There is a variable speed drive serving the fan motor.  The cooling coil
is served by chilled water produced by a chiller and the heating coil is
served by hot water produced by a gas boiler.
</p>

<p>
<br>
</p>

<p>
<img src=\"../../../doc/images/Schematic.png\"/>
<figcaption><small>Figure 1: System schematic.</small></figcaption>
</p>

<p>
<br>
</p>

<h4>Equipment specifications and performance maps</h4>
<p>
For the fan, the design airflow rate is 0.55 kg/s and design pressure rise is
185 Pa.  The fan and motor efficiencies are both constant at 0.7.
The heat from the motor is added to the air stream.

The COP of the chiller is assumed constant at 3.0.  The efficiency of the
gas boiler is assumed constant at 0.9.
</p>
<h4>Rule-based or local-loop controllers (if included)</h4>
<p>
A baseline thermostat controller provides heating and cooling as necessary
to the room by modulating the supply air temperature and
fan speed.  The thermostat, designated as C1 in Figure 1 and shown in Figure 2 below,
uses two different PI controllers for heating and
cooling, each taking the respective zone temperature set point and zone
temperature measurement as inputs.  The outputs are used to control supply air
temperature set point and fan speed according to the map shown in Figure 3 below.
The supply air temperature is exactly met by the coils using an ideal controller
depicted as C2 in Figure 1.
For heating, the maximum supply air temperature is 40 C and the minimum is the
zone occupied heating temperature setpoint.  For cooling, the minimum supply
air temperature is 12 C and the maximum is the zone occupied cooling
temperature setpoint.
</p>

<p>
<br>
</p>

<p>
<img src=\"../../../doc/images/C1.png\"/>
<figcaption><small>Figure 2: Controller C1.</small></figcaption>
</p>

<p>
<br>
</p>

<p>
<img src=\"../../../doc/images/ControlSchematic_Ideal.png\" width=600 />
<figcaption><small>Figure 3: Mapping of PI output to supply air temperature set point and fan speed in controller C1.</small></figcaption>
</p>

<p>
<br>
</p>

<h3>Model IO's</h3>
<h4>Inputs</h4>
The model inputs are:
<ul>
<li>
<code>fcu_oveTSup_u</code> [K] [min=285.15, max=313.15]: Supply air temperature setpoint
</li>
<li>
<code>fcu_oveFan_u</code> [1] [min=0.0, max=1.0]: Fan control signal as air mass flow rate normalized to the design air mass flow rate
</li>
<li>
<code>con_oveTSetHea_u</code> [K] [min=288.15, max=296.15]: Zone temperature setpoint for heating
</li>
<li>
<code>con_oveTSetCoo_u</code> [K] [min=296.15, max=303.15]: Zone temperature setpoint for cooling
</li>
</ul>
<h4>Outputs</h4>
The model outputs are:
<ul>
<li>
<code>fcu_reaFloSup_y</code> [kg/s] [min=None, max=None]: Supply air mass flow rate
</li>
<li>
<code>fcu_reaPCoo_y</code> [W] [min=None, max=None]: Cooling electrical power consumption
</li>
<li>
<code>fcu_reaPFan_y</code> [W] [min=None, max=None]: Supply fan electrical power consumption
</li>
<li>
<code>fcu_reaPHea_y</code> [W] [min=None, max=None]: Heating thermal power consumption
</li>
<li>
<code>zon_reaCO2RooAir_y</code> [ppm] [min=None, max=None]: Zone air CO2 concentration
</li>
<li>
<code>zon_reaPLig_y</code> [W] [min=None, max=None]: Lighting power submeter
</li>
<li>
<code>zon_reaPPlu_y</code> [W] [min=None, max=None]: Plug load power submeter
</li>
<li>
<code>zon_reaTRooAir_y</code> [K] [min=None, max=None]: Zone air temperature
</li>
<li>
<code>zon_weaSta_reaWeaCeiHei_y</code> [m] [min=None, max=None]: Cloud cover ceiling height measurement
</li>
<li>
<code>zon_weaSta_reaWeaCloTim_y</code> [s] [min=None, max=None]: Day number with units of seconds
</li>
<li>
<code>zon_weaSta_reaWeaHDifHor_y</code> [W/m2] [min=None, max=None]: Horizontal diffuse solar radiation measurement
</li>
<li>
<code>zon_weaSta_reaWeaHDirNor_y</code> [W/m2] [min=None, max=None]: Direct normal radiation measurement
</li>
<li>
<code>zon_weaSta_reaWeaHGloHor_y</code> [W/m2] [min=None, max=None]: Global horizontal solar irradiation measurement
</li>
<li>
<code>zon_weaSta_reaWeaHHorIR_y</code> [W/m2] [min=None, max=None]: Horizontal infrared irradiation measurement
</li>
<li>
<code>zon_weaSta_reaWeaLat_y</code> [rad] [min=None, max=None]: Latitude of the location
</li>
<li>
<code>zon_weaSta_reaWeaLon_y</code> [rad] [min=None, max=None]: Longitude of the location
</li>
<li>
<code>zon_weaSta_reaWeaNOpa_y</code> [1] [min=None, max=None]: Opaque sky cover measurement
</li>
<li>
<code>zon_weaSta_reaWeaNTot_y</code> [1] [min=None, max=None]: Sky cover measurement
</li>
<li>
<code>zon_weaSta_reaWeaPAtm_y</code> [Pa] [min=None, max=None]: Atmospheric pressure measurement
</li>
<li>
<code>zon_weaSta_reaWeaRelHum_y</code> [1] [min=None, max=None]: Outside relative humidity measurement
</li>
<li>
<code>zon_weaSta_reaWeaSolAlt_y</code> [rad] [min=None, max=None]: Solar altitude angle measurement
</li>
<li>
<code>zon_weaSta_reaWeaSolDec_y</code> [rad] [min=None, max=None]: Solar declination angle measurement
</li>
<li>
<code>zon_weaSta_reaWeaSolHouAng_y</code> [rad] [min=None, max=None]: Solar hour angle measurement
</li>
<li>
<code>zon_weaSta_reaWeaSolTim_y</code> [s] [min=None, max=None]: Solar time
</li>
<li>
<code>zon_weaSta_reaWeaSolZen_y</code> [rad] [min=None, max=None]: Solar zenith angle measurement
</li>
<li>
<code>zon_weaSta_reaWeaTBlaSky_y</code> [K] [min=None, max=None]: Black-body sky temperature measurement
</li>
<li>
<code>zon_weaSta_reaWeaTDewPoi_y</code> [K] [min=None, max=None]: Dew point temperature measurement
</li>
<li>
<code>zon_weaSta_reaWeaTDryBul_y</code> [K] [min=None, max=None]: Outside drybulb temperature measurement
</li>
<li>
<code>zon_weaSta_reaWeaTWetBul_y</code> [K] [min=None, max=None]: Wet bulb temperature measurement
</li>
<li>
<code>zon_weaSta_reaWeaWinDir_y</code> [rad] [min=None, max=None]: Wind direction measurement
</ul>
<h3>Additional System Design</h3>
<h4>Lighting</h4>
<p>
Artificial lighting is provided by hanging fluorescent fixtures.
</p>
<h4>Shading</h4>
<p>
There are no shades on the building.
</p>
<h4>Onsite Generation and Storage</h4>
<p>
There is no energy generation or storage on the site.
</p>
<h3>Model Implementation Details</h3>
<h4>Moist vs. dry air</h4>
<p>
A moist air model is used, but condensation is not modeled on the cooling coil
and humidity is not monitored.

</p>
<h4>Pressure-flow models</h4>
<p>
The FCU fan is speed-controlled and the resulting flow is calculated based
on resulting pressure rise by the fan and fixed pressure drop of the system.
</p>
<h4>Infiltration models</h4>
<p>
A constant infiltration flowrate is assumed to be 0.5 ACH.
</p>
<h4>Other assumptions</h4>
<p>
The supply air temperature is directly specified.
</p>
<p>
CO2 generation is 0.0048 L/s per person (Table 5, Persily and De Jonge 2017)
and density of CO2 assumed to be 1.8 kg/m^3,
making CO2 generation 8.64e-6 kg/s per person.
Outside air CO2 concentration is 400 ppm.  However, CO2 concentration
is not controlled for in the model.
</p>
<p>
Persily, A. and De Jonge, L. (2017).
Carbon dioxide generation rates for building occupants.
Indoor Air, 27, 868–879.  https://doi.org/10.1111/ina.12383.
</p>
<h3>Scenario Information</h3>
<h4>Time Periods</h4>
<p>
The <b>Peak Heat Day</b> (specifier for <code>/scenario</code> API is <code>'peak_heat_day'</code>) period is:
<ul>
This testing time period is a two-week test with one-week warmup period utilizing
baseline control.  The two-week period is centered on the day with the
maximum 15-minute system heating load in the year.
</ul>
<ul>
Start Time: Day 334.
</ul>
<ul>
End Time: Day 348.
</ul>
</p>
<p>
The <b>Typical Heat Day</b> (specifier for <code>/scenario</code> API is <code>'typical_heat_day'</code>) period is:
<ul>
This testing time period is a two-week test with one-week warmup period utilizing
baseline control.  The two-week period is centered on the day with day with
the maximum 15-minute system heating load that is closest from below to the
median of all 15-minute maximum heating loads of all days in the year.
</ul>
<ul>
Start Time: Day 44.
</ul>
<ul>
End Time: Day 58.
</ul>
</p>
<p>
The <b>Peak Cool Day</b> (specifier for <code>/scenario</code> API is <code>'peak_cool_day'</code>) period is:
<ul>
This testing time period is a two-week test with one-week warmup period utilizing
baseline control.  The two-week period is centered on the day with the
maximum 15-minute system cooling load in the year.
</ul>
<ul>
Start Time: Day 282.
</ul>
<ul>
End Time: Day 296.
</ul>
</p>
<p>
The <b>Typical Cool Day</b> (specifier for <code>/scenario</code> API is <code>'typical_cool_day'</code>) period is:
<ul>
This testing time period is a two-week test with one-week warmup period utilizing
baseline control.  The two-week period is centered on the day with day with
the maximum 15-minute system cooling load that is closest from below to the
median of all 15-minute maximum cooling loads of all days in the year.
</ul>
<ul>
Start Time: Day 146.
</ul>
<ul>
End Time: Day 160.
</ul>
</p>
<p>
The <b>Mix Day</b> (specifier for <code>/scenario</code> API is <code>'mix_day'</code>) period is:
<ul>
This testing time period is a two-week test with one-week warmup period utilizing
baseline control.  The two-week period is centered on the day with the maximimum
sum of daily heating and cooling loads minus the difference between
daily heating and cooling loads.  This is a day with both significant heating
and cooling loads.
</ul>
<ul>
Start Time: Day 14.
</ul>
<ul>
End Time: Day 28.
</ul>
</p>
<h4>Energy Pricing</h4>
<p>
The <b>Constant Electricity Price</b> (specifier for <code>/scenario</code> API is <code>'constant'</code>) profile is:
<ul>
Based on the Schedule R tariff
for winter season and summer season first 500 kWh as defined by the
utility servicing the assumed location of the test case.  It is $0.05461/kWh.
For reference,
see https://www.xcelenergy.com/company/rates_and_regulations/rates/rate_books
in the section on Current Tariffs/Electric Rate Books (PDF).
</ul>
<ul>
Specifier for <code>/scenario</code> API is <code>'constant'</code>.
</ul>
</p>
<p>
The <b>Dynamic Electricity Price</b> (specifier for <code>/scenario</code> API is <code>'dynamic'</code>) profile is:
<ul>
Based on the Schedule RE-TOU tariff
as defined by the utility servicing the assumed location of the test case.
For reference,
see https://www.xcelenergy.com/company/rates_and_regulations/rates/rate_books
in the section on Current Tariffs/Electric Rate Books (PDF).
</ul>
</p>
<p>
<ul>
<li>
Summer on-peak is $0.13814/kWh.
</li>
<li>
Summer mid-peak is $0.08420/kWh.
</li>
<li>
Summer off-peak is $0.04440/kWh.
</li>
<li>
Winter on-peak is $0.08880/kWh.
</li>
<li>
Winter mid-peak is $0.05413/kWh.
</li>
<li>
Winter off-peak is $0.04440/kWh.
</li>
<li>
The Summer season is June 1 to September 30.
</li>
<li>
The Winter season is October 1 to May 31.
</li>
</p>
<p>
<u>The On-Peak Period is</u>:
<ul>
<li>
Summer and Winter weekdays except Holidays, between 2:00 p.m. and 6:00 p.m.
local time.
</li>
</ul>
<u>The Mid-Peak Period is</u>:
<ul>
<li>
Summer and Winter weekdays except Holidays, between 9:00 a.m. and
2:00 p.m. and between 6:00 p.m. and 9:00 p.m. local time.
</li>
<li>
Summer and Winter weekends and Holidays, between 9:00 a.m. and
9:00 p.m. local time.
</li>
</ul>
<u>The Off-Peak Period is</u>:
<ul>
<li>
Summer and Winter daily, between 9:00 p.m. and 9:00 a.m. local time.
</li>
</ul>
</ul>
</p>
<p>
The <b>Highly Dynamic Electricity Price</b> (specifier for <code>/scenario</code> API is <code>'highly_dynamic'</code>) profile is:
<ul>
Based on the the
day-ahead energy prices (LMP) as determined in the Southwest Power Pool
wholesale electricity market for node LAM345 in the year 2018.
For reference,
see https://marketplace.spp.org/pages/da-lmp-by-location#%2F2018.
</ul>
</p>
<p>
The <b>Gas Price</b> profile is:
<ul>
Based on the Schedule R tariff for usage price per therm as defined by the
utility servicing the assumed location of the test case.  It is $0.002878/kWh
($0.0844/therm).
For reference,
see https://www.xcelenergy.com/company/rates_and_regulations/rates/rate_books
in the section on Summary of Gas Rates for 10/1/19.
</ul>
</p>
<h4>Emission Factors</h4>
<p>
The <b>Electricity Emissions Factor</b> profile is:
<ul>
Based on the average electricity generation mix for CO,USA for the year of
2017.  It is 0.6618 kgCO2/kWh (1459 lbsCO2/MWh).
For reference,
see https://www.eia.gov/electricity/state/colorado/.
</ul>
</p>
<p>
The <b>Gas Emissions Factor</b> profile is:
<ul>
Based on the kgCO2 emitted per amount of natural gas burned in terms of
energy content.  It is 0.18108 kgCO2/kWh (53.07 kgCO2/milBTU).
For reference,
see https://www.eia.gov/environment/emissions/co2_vol_mass.php.
</ul>
</p>
</html>",
    revisions="<html>
<ul>
<li>
December 6, 2021, by David Blum:<br/>
Correct mix day time period.
This is for <a href=https://github.com/ibpsa/project1-boptest/issues/381>
BOPTEST issue #381</a>.
</li>
<li>
April 13, 2021, by David Blum:<br/>
Add time period documentation.
</li>
<li>
November 10, 2020, by David Blum:<br/>
Add weather station measurements.
</li>
<li>
March 4, 2020, by David Blum:<br/>
Updated CO2 generation per person and method of ppm calculation.
</li>
<li>
December 15, 2019, by David Blum:<br/>
First implementation.
</li>
</ul>
</html>"));
    end Baseline;

    model LoadShifting "Testcase model with ideal airflow"
      extends Modelica.Icons.Example;
      BaseClasses.Case900FF zon(mAir_flow_nominal=fcu.mAir_flow_nominal)
        annotation (Placement(transformation(extent={{34,-10},{54,10}})));

      BaseClasses.Thermostat_T con "Thermostat controller"
        annotation (Placement(transformation(extent={{-80,-10},{-60,10}})));
      BaseClasses.FanCoilUnit_T fcu "Fan coil unit"
        annotation (Placement(transformation(extent={{-20,-8},{0,20}})));
      Modelica.Blocks.Logical.Switch swi
        annotation (Placement(transformation(extent={{-40,-50},{-20,-70}})));
      Modelica.Blocks.Sources.Constant TSupAirSet(k=273.15 + 14)
        annotation (Placement(transformation(extent={{-80,40},{-60,60}})));
      Modelica.Blocks.Sources.CombiTimeTable uFanOve(
        smoothness=Modelica.Blocks.Types.Smoothness.ConstantSegments,
        extrapolation=Modelica.Blocks.Types.Extrapolation.Periodic,
        table=[0,0; 12*3600,0.5; 13*3600,0.2; 24*3600,0])
        annotation (Placement(transformation(extent={{-80,-100},{-60,-80}})));
      Modelica.Blocks.Sources.BooleanTable uFanOveAct(table={200*24*3600 + 12*
            3600,200*24*3600 + 15*3600})
        annotation (Placement(transformation(extent={{-80,-70},{-60,-50}})));
    equation
      connect(fcu.supplyAir, zon.supplyAir) annotation (Line(points={{0,13.7778},
              {20,13.7778},{20,2},{34,2}},color={0,127,255}));
      connect(fcu.returnAir, zon.returnAir) annotation (Line(points={{0,-6.44444},{
              20,-6.44444},{20,-2},{34,-2}}, color={0,127,255}));
      connect(zon.TRooAir, con.TZon) annotation (Line(points={{61,0},{80,0},{80,-40},
              {-100,-40},{-100,0},{-82,0}}, color={0,0,127}));
      connect(TSupAirSet.y, fcu.TSup) annotation (Line(points={{-59,50},{-40,50},
              {-40,9.11111},{-21.4286,9.11111}}, color={0,0,127}));
      connect(con.yFan, swi.u3) annotation (Line(points={{-59,0},{-52,0},{-52,
              -52},{-42,-52}}, color={0,0,127}));
      connect(swi.y, fcu.uFan) annotation (Line(points={{-19,-60},{-4,-60},{-4,
              -24},{-40,-24},{-40,2.88889},{-21.4286,2.88889}}, color={0,0,127}));
      connect(uFanOve.y[1], swi.u1) annotation (Line(points={{-59,-90},{-52,-90},
              {-52,-68},{-42,-68}}, color={0,0,127}));
      connect(uFanOveAct.y, swi.u2)
        annotation (Line(points={{-59,-60},{-42,-60}}, color={255,0,255}));
      annotation (Icon(coordinateSystem(preserveAspectRatio=false)), Diagram(
            coordinateSystem(preserveAspectRatio=false)),
        experiment(
          StartTime=17280000,
          StopTime=17366400,
          Interval=300,
          Tolerance=1e-06,
          __Dymola_Algorithm="Cvode"),
    Documentation(info="<html>
General model description.
<h3>Building Design and Use</h3>
<h4>Architecture</h4>
<p>
The building is a single room based on the BESTEST Case 900 model definition.
The floor dimensions are 6m x 8m and the floor-to-ceiling height is 2.7m.
There are four exterior walls facing the cardinal directions and a flat roof.
The walls facing east-west have the short dimension.  The south wall contains
two windows, each 3m wide and 2m tall.  The use of the building is assumed
to be a two-person office with a light load density.
</p>
<h4>Constructions</h4>
<p>
The constructions are based on the BESTEST Case 900 model definition.  The
exterior walls are made of concrete block and insulation, while the floor
is a concrete slab.  The roof is made of wood frame with insulation.  The
layer-by-layer specifications are (Outside to Inside):
</p>
<p>
<b>Exterior Walls</b>
<table>
  <tr>
  <th>Name</th>
  <th>Thickness [m]</th>
  <th>Thermal Conductivity [W/m-K]</th>
  <th>Specific Heat Capacity [J/kg-K]</th>
  <th>Density [kg/m3]</th>
  </tr>
  <tr>
  <td>Layer 1</td>
  <td>0.009</td>
  <td>0.140</td>
  <td>900</td>
  <td>530</td>
  </tr>
  <tr>
  <td>Layer 2</td>
  <td>0.0615</td>
  <td>0.040</td>
  <td>1400</td>
  <td>10</td>
  </tr>
  <tr>
  <td>Layer 3</td>
  <td>0.100</td>
  <td>0.510</td>
  <td>1000</td>
  <td>1400</td>
  </tr>
  </table>
<table>
  <tr>
  <th>Name</th>
  <th>IR Emissivity [-]</th>
  <th>Solar Emissivity [-]</th>
  </tr>
  <tr>
  <td>Outside</td>
  <td>0.9</td>
  <td>0.6</td>
  </tr>
  <tr>
  <td>Inside</td>
  <td>0.9</td>
  <td>0.6</td>
  </tr>
</table>
</p>
<p>
<b>Floor</b>
<table>
  <tr>
  <th>Name</th>
  <th>Thickness [m]</th>
  <th>Thermal Conductivity [W/m-K]</th>
  <th>Specific Heat Capacity [J/kg-K]</th>
  <th>Density [kg/m3]</th>
  </tr>
  <tr>
  <td>Layer 1</td>
  <td>1.007</td>
  <td>0.040</td>
  <td>0</td>
  <td>0</td>
  </tr>
  <tr>
  <td>Layer 2</td>
  <td>0.080</td>
  <td>1.130</td>
  <td>1000</td>
  <td>1400</td>
  </tr>
</table>
<table>
  <tr>
  <th>Name</th>
  <th>IR Emissivity [-]</th>
  <th>Solar Emissivity [-]</th>
  </tr>
  <tr>
  <td>Outside</td>
  <td>0.9</td>
  <td>0.6</td>
  </tr>
  <tr>
  <td>Inside</td>
  <td>0.9</td>
  <td>0.6</td>
  </tr>
</table>
</p>
<p>
<b>Roof</b>
<table>
  <tr>
  <th>Name</th>
  <th>Thickness [m]</th>
  <th>Thermal Conductivity [W/m-K]</th>
  <th>Specific Heat Capacity [J/kg-K]</th>
  <th>Density [kg/m3]</th>
  </tr>
  <tr>
  <td>Layer 1</td>
  <td>0.019</td>
  <td>0.140</td>
  <td>900</td>
  <td>530</td>
  </tr>
  <tr>
  <td>Layer 2</td>
  <td>0.1118</td>
  <td>0.040</td>
  <td>840</td>
  <td>12</td>
  </tr>
  <tr>
  <td>Layer 3</td>
  <td>0.010</td>
  <td>0.160</td>
  <td>840</td>
  <td>950</td>
  </tr>
</table>
<table>
  <tr>
  <th>Name</th>
  <th>IR Emissivity [-]</th>
  <th>Solar Emissivity [-]</th>
  </tr>
  <tr>
  <td>Outside</td>
  <td>0.9</td>
  <td>0.6</td>
  </tr>
  <tr>
  <td>Inside</td>
  <td>0.9</td>
  <td>0.6</td>
  </tr>
</table>
<p>
The windows are double pane clear 3.175mm glass with 13mm air gap.
</p>

<h4>Occupancy schedules</h4>
<p>
There is maximum occupancy (two people) from 8am to 6pm each day,
and no occupancy during all other times.
</p>
<h4>Internal loads and schedules</h4>
<p>
The internal heat gains from plug loads come mainly from computers and monitors.
The internal heat gains from lighting come from hanging fluorescent fixtures.
Both types of loads are at maximum during occupied periods and 0.1 maximum
during all other times.  The occupied heating and cooling temperature
setpoints are 21 C and 24 C respectively, while the unoccupied heating
and cooling temperature setpoints are 15 C and 30 C respectively.

</p>
<h4>Climate data</h4>
<p>
The climate is assumed to be near Denver, CO, USA with a latitude and
longitude of 39.76,-104.86.  The climate data comes from the
Denver-Stapleton,CO,USA,TMY.
</p>
<h3>HVAC System Design</h3>
<h4>Primary and secondary system designs</h4>
<p>
Heating and cooling is provided to the office using an idealized four-pipe
fan coil unit (FCU), presented in Figure 1 below.
The FCU contains a fan, cooling coil, heating coil,
and filter.  The fan draws room air into the unit, blows it over the coils
and through the filter, and supplies the conditioned air back to the room.
There is a variable speed drive serving the fan motor.  The cooling coil
is served by chilled water produced by a chiller and the heating coil is
served by hot water produced by a gas boiler.
</p>

<p>
<br>
</p>

<p>
<img src=\"../../../doc/images/Schematic.png\"/>
<figcaption><small>Figure 1: System schematic.</small></figcaption>
</p>

<p>
<br>
</p>

<h4>Equipment specifications and performance maps</h4>
<p>
For the fan, the design airflow rate is 0.55 kg/s and design pressure rise is
185 Pa.  The fan and motor efficiencies are both constant at 0.7.
The heat from the motor is added to the air stream.

The COP of the chiller is assumed constant at 3.0.  The efficiency of the
gas boiler is assumed constant at 0.9.
</p>
<h4>Rule-based or local-loop controllers (if included)</h4>
<p>
A baseline thermostat controller provides heating and cooling as necessary
to the room by modulating the supply air temperature and
fan speed.  The thermostat, designated as C1 in Figure 1 and shown in Figure 2 below,
uses two different PI controllers for heating and
cooling, each taking the respective zone temperature set point and zone
temperature measurement as inputs.  The outputs are used to control supply air
temperature set point and fan speed according to the map shown in Figure 3 below.
The supply air temperature is exactly met by the coils using an ideal controller
depicted as C2 in Figure 1.
For heating, the maximum supply air temperature is 40 C and the minimum is the
zone occupied heating temperature setpoint.  For cooling, the minimum supply
air temperature is 12 C and the maximum is the zone occupied cooling
temperature setpoint.
</p>

<p>
<br>
</p>

<p>
<img src=\"../../../doc/images/C1.png\"/>
<figcaption><small>Figure 2: Controller C1.</small></figcaption>
</p>

<p>
<br>
</p>

<p>
<img src=\"../../../doc/images/ControlSchematic_Ideal.png\" width=600 />
<figcaption><small>Figure 3: Mapping of PI output to supply air temperature set point and fan speed in controller C1.</small></figcaption>
</p>

<p>
<br>
</p>

<h3>Model IO's</h3>
<h4>Inputs</h4>
The model inputs are:
<ul>
<li>
<code>fcu_oveTSup_u</code> [K] [min=285.15, max=313.15]: Supply air temperature setpoint
</li>
<li>
<code>fcu_oveFan_u</code> [1] [min=0.0, max=1.0]: Fan control signal as air mass flow rate normalized to the design air mass flow rate
</li>
<li>
<code>con_oveTSetHea_u</code> [K] [min=288.15, max=296.15]: Zone temperature setpoint for heating
</li>
<li>
<code>con_oveTSetCoo_u</code> [K] [min=296.15, max=303.15]: Zone temperature setpoint for cooling
</li>
</ul>
<h4>Outputs</h4>
The model outputs are:
<ul>
<li>
<code>fcu_reaFloSup_y</code> [kg/s] [min=None, max=None]: Supply air mass flow rate
</li>
<li>
<code>fcu_reaPCoo_y</code> [W] [min=None, max=None]: Cooling electrical power consumption
</li>
<li>
<code>fcu_reaPFan_y</code> [W] [min=None, max=None]: Supply fan electrical power consumption
</li>
<li>
<code>fcu_reaPHea_y</code> [W] [min=None, max=None]: Heating thermal power consumption
</li>
<li>
<code>zon_reaCO2RooAir_y</code> [ppm] [min=None, max=None]: Zone air CO2 concentration
</li>
<li>
<code>zon_reaPLig_y</code> [W] [min=None, max=None]: Lighting power submeter
</li>
<li>
<code>zon_reaPPlu_y</code> [W] [min=None, max=None]: Plug load power submeter
</li>
<li>
<code>zon_reaTRooAir_y</code> [K] [min=None, max=None]: Zone air temperature
</li>
<li>
<code>zon_weaSta_reaWeaCeiHei_y</code> [m] [min=None, max=None]: Cloud cover ceiling height measurement
</li>
<li>
<code>zon_weaSta_reaWeaCloTim_y</code> [s] [min=None, max=None]: Day number with units of seconds
</li>
<li>
<code>zon_weaSta_reaWeaHDifHor_y</code> [W/m2] [min=None, max=None]: Horizontal diffuse solar radiation measurement
</li>
<li>
<code>zon_weaSta_reaWeaHDirNor_y</code> [W/m2] [min=None, max=None]: Direct normal radiation measurement
</li>
<li>
<code>zon_weaSta_reaWeaHGloHor_y</code> [W/m2] [min=None, max=None]: Global horizontal solar irradiation measurement
</li>
<li>
<code>zon_weaSta_reaWeaHHorIR_y</code> [W/m2] [min=None, max=None]: Horizontal infrared irradiation measurement
</li>
<li>
<code>zon_weaSta_reaWeaLat_y</code> [rad] [min=None, max=None]: Latitude of the location
</li>
<li>
<code>zon_weaSta_reaWeaLon_y</code> [rad] [min=None, max=None]: Longitude of the location
</li>
<li>
<code>zon_weaSta_reaWeaNOpa_y</code> [1] [min=None, max=None]: Opaque sky cover measurement
</li>
<li>
<code>zon_weaSta_reaWeaNTot_y</code> [1] [min=None, max=None]: Sky cover measurement
</li>
<li>
<code>zon_weaSta_reaWeaPAtm_y</code> [Pa] [min=None, max=None]: Atmospheric pressure measurement
</li>
<li>
<code>zon_weaSta_reaWeaRelHum_y</code> [1] [min=None, max=None]: Outside relative humidity measurement
</li>
<li>
<code>zon_weaSta_reaWeaSolAlt_y</code> [rad] [min=None, max=None]: Solar altitude angle measurement
</li>
<li>
<code>zon_weaSta_reaWeaSolDec_y</code> [rad] [min=None, max=None]: Solar declination angle measurement
</li>
<li>
<code>zon_weaSta_reaWeaSolHouAng_y</code> [rad] [min=None, max=None]: Solar hour angle measurement
</li>
<li>
<code>zon_weaSta_reaWeaSolTim_y</code> [s] [min=None, max=None]: Solar time
</li>
<li>
<code>zon_weaSta_reaWeaSolZen_y</code> [rad] [min=None, max=None]: Solar zenith angle measurement
</li>
<li>
<code>zon_weaSta_reaWeaTBlaSky_y</code> [K] [min=None, max=None]: Black-body sky temperature measurement
</li>
<li>
<code>zon_weaSta_reaWeaTDewPoi_y</code> [K] [min=None, max=None]: Dew point temperature measurement
</li>
<li>
<code>zon_weaSta_reaWeaTDryBul_y</code> [K] [min=None, max=None]: Outside drybulb temperature measurement
</li>
<li>
<code>zon_weaSta_reaWeaTWetBul_y</code> [K] [min=None, max=None]: Wet bulb temperature measurement
</li>
<li>
<code>zon_weaSta_reaWeaWinDir_y</code> [rad] [min=None, max=None]: Wind direction measurement
</ul>
<h3>Additional System Design</h3>
<h4>Lighting</h4>
<p>
Artificial lighting is provided by hanging fluorescent fixtures.
</p>
<h4>Shading</h4>
<p>
There are no shades on the building.
</p>
<h4>Onsite Generation and Storage</h4>
<p>
There is no energy generation or storage on the site.
</p>
<h3>Model Implementation Details</h3>
<h4>Moist vs. dry air</h4>
<p>
A moist air model is used, but condensation is not modeled on the cooling coil
and humidity is not monitored.

</p>
<h4>Pressure-flow models</h4>
<p>
The FCU fan is speed-controlled and the resulting flow is calculated based
on resulting pressure rise by the fan and fixed pressure drop of the system.
</p>
<h4>Infiltration models</h4>
<p>
A constant infiltration flowrate is assumed to be 0.5 ACH.
</p>
<h4>Other assumptions</h4>
<p>
The supply air temperature is directly specified.
</p>
<p>
CO2 generation is 0.0048 L/s per person (Table 5, Persily and De Jonge 2017)
and density of CO2 assumed to be 1.8 kg/m^3,
making CO2 generation 8.64e-6 kg/s per person.
Outside air CO2 concentration is 400 ppm.  However, CO2 concentration
is not controlled for in the model.
</p>
<p>
Persily, A. and De Jonge, L. (2017).
Carbon dioxide generation rates for building occupants.
Indoor Air, 27, 868–879.  https://doi.org/10.1111/ina.12383.
</p>
<h3>Scenario Information</h3>
<h4>Time Periods</h4>
<p>
The <b>Peak Heat Day</b> (specifier for <code>/scenario</code> API is <code>'peak_heat_day'</code>) period is:
<ul>
This testing time period is a two-week test with one-week warmup period utilizing
baseline control.  The two-week period is centered on the day with the
maximum 15-minute system heating load in the year.
</ul>
<ul>
Start Time: Day 334.
</ul>
<ul>
End Time: Day 348.
</ul>
</p>
<p>
The <b>Typical Heat Day</b> (specifier for <code>/scenario</code> API is <code>'typical_heat_day'</code>) period is:
<ul>
This testing time period is a two-week test with one-week warmup period utilizing
baseline control.  The two-week period is centered on the day with day with
the maximum 15-minute system heating load that is closest from below to the
median of all 15-minute maximum heating loads of all days in the year.
</ul>
<ul>
Start Time: Day 44.
</ul>
<ul>
End Time: Day 58.
</ul>
</p>
<p>
The <b>Peak Cool Day</b> (specifier for <code>/scenario</code> API is <code>'peak_cool_day'</code>) period is:
<ul>
This testing time period is a two-week test with one-week warmup period utilizing
baseline control.  The two-week period is centered on the day with the
maximum 15-minute system cooling load in the year.
</ul>
<ul>
Start Time: Day 282.
</ul>
<ul>
End Time: Day 296.
</ul>
</p>
<p>
The <b>Typical Cool Day</b> (specifier for <code>/scenario</code> API is <code>'typical_cool_day'</code>) period is:
<ul>
This testing time period is a two-week test with one-week warmup period utilizing
baseline control.  The two-week period is centered on the day with day with
the maximum 15-minute system cooling load that is closest from below to the
median of all 15-minute maximum cooling loads of all days in the year.
</ul>
<ul>
Start Time: Day 146.
</ul>
<ul>
End Time: Day 160.
</ul>
</p>
<p>
The <b>Mix Day</b> (specifier for <code>/scenario</code> API is <code>'mix_day'</code>) period is:
<ul>
This testing time period is a two-week test with one-week warmup period utilizing
baseline control.  The two-week period is centered on the day with the maximimum
sum of daily heating and cooling loads minus the difference between
daily heating and cooling loads.  This is a day with both significant heating
and cooling loads.
</ul>
<ul>
Start Time: Day 14.
</ul>
<ul>
End Time: Day 28.
</ul>
</p>
<h4>Energy Pricing</h4>
<p>
The <b>Constant Electricity Price</b> (specifier for <code>/scenario</code> API is <code>'constant'</code>) profile is:
<ul>
Based on the Schedule R tariff
for winter season and summer season first 500 kWh as defined by the
utility servicing the assumed location of the test case.  It is $0.05461/kWh.
For reference,
see https://www.xcelenergy.com/company/rates_and_regulations/rates/rate_books
in the section on Current Tariffs/Electric Rate Books (PDF).
</ul>
<ul>
Specifier for <code>/scenario</code> API is <code>'constant'</code>.
</ul>
</p>
<p>
The <b>Dynamic Electricity Price</b> (specifier for <code>/scenario</code> API is <code>'dynamic'</code>) profile is:
<ul>
Based on the Schedule RE-TOU tariff
as defined by the utility servicing the assumed location of the test case.
For reference,
see https://www.xcelenergy.com/company/rates_and_regulations/rates/rate_books
in the section on Current Tariffs/Electric Rate Books (PDF).
</ul>
</p>
<p>
<ul>
<li>
Summer on-peak is $0.13814/kWh.
</li>
<li>
Summer mid-peak is $0.08420/kWh.
</li>
<li>
Summer off-peak is $0.04440/kWh.
</li>
<li>
Winter on-peak is $0.08880/kWh.
</li>
<li>
Winter mid-peak is $0.05413/kWh.
</li>
<li>
Winter off-peak is $0.04440/kWh.
</li>
<li>
The Summer season is June 1 to September 30.
</li>
<li>
The Winter season is October 1 to May 31.
</li>
</p>
<p>
<u>The On-Peak Period is</u>:
<ul>
<li>
Summer and Winter weekdays except Holidays, between 2:00 p.m. and 6:00 p.m.
local time.
</li>
</ul>
<u>The Mid-Peak Period is</u>:
<ul>
<li>
Summer and Winter weekdays except Holidays, between 9:00 a.m. and
2:00 p.m. and between 6:00 p.m. and 9:00 p.m. local time.
</li>
<li>
Summer and Winter weekends and Holidays, between 9:00 a.m. and
9:00 p.m. local time.
</li>
</ul>
<u>The Off-Peak Period is</u>:
<ul>
<li>
Summer and Winter daily, between 9:00 p.m. and 9:00 a.m. local time.
</li>
</ul>
</ul>
</p>
<p>
The <b>Highly Dynamic Electricity Price</b> (specifier for <code>/scenario</code> API is <code>'highly_dynamic'</code>) profile is:
<ul>
Based on the the
day-ahead energy prices (LMP) as determined in the Southwest Power Pool
wholesale electricity market for node LAM345 in the year 2018.
For reference,
see https://marketplace.spp.org/pages/da-lmp-by-location#%2F2018.
</ul>
</p>
<p>
The <b>Gas Price</b> profile is:
<ul>
Based on the Schedule R tariff for usage price per therm as defined by the
utility servicing the assumed location of the test case.  It is $0.002878/kWh
($0.0844/therm).
For reference,
see https://www.xcelenergy.com/company/rates_and_regulations/rates/rate_books
in the section on Summary of Gas Rates for 10/1/19.
</ul>
</p>
<h4>Emission Factors</h4>
<p>
The <b>Electricity Emissions Factor</b> profile is:
<ul>
Based on the average electricity generation mix for CO,USA for the year of
2017.  It is 0.6618 kgCO2/kWh (1459 lbsCO2/MWh).
For reference,
see https://www.eia.gov/electricity/state/colorado/.
</ul>
</p>
<p>
The <b>Gas Emissions Factor</b> profile is:
<ul>
Based on the kgCO2 emitted per amount of natural gas burned in terms of
energy content.  It is 0.18108 kgCO2/kWh (53.07 kgCO2/milBTU).
For reference,
see https://www.eia.gov/environment/emissions/co2_vol_mass.php.
</ul>
</p>
</html>",
    revisions="<html>
<ul>
<li>
December 6, 2021, by David Blum:<br/>
Correct mix day time period.
This is for <a href=https://github.com/ibpsa/project1-boptest/issues/381>
BOPTEST issue #381</a>.
</li>
<li>
April 13, 2021, by David Blum:<br/>
Add time period documentation.
</li>
<li>
November 10, 2020, by David Blum:<br/>
Add weather station measurements.
</li>
<li>
March 4, 2020, by David Blum:<br/>
Updated CO2 generation per person and method of ppm calculation.
</li>
<li>
December 15, 2019, by David Blum:<br/>
First implementation.
</li>
</ul>
</html>"));
    end LoadShifting;
  end TestCases;

  package BaseClasses "Base classes for simple air testcase."
    extends Modelica.Icons.BasesPackage;
    model Case900FF "Case 600FF, but with high thermal mass"
      extends Case600FF(
        matExtWal=extWalCase900,
        matFlo=floorCase900,
        staRes(
          minT(
            Min=-6.4 + 273.15,
            Max=-1.6 + 273.15,
            Mean=-4.2 + 273.15),
          maxT(
            Min=41.6 + 273.15,
            Max=44.8 + 273.15,
            Mean=43.1 + 273.15),
          meanT(
            Min=24.5 + 273.15,
            Max=25.9 + 273.15,
            Mean=25.2 + 273.15)));

      parameter Buildings.ThermalZones.Detailed.Validation.BESTEST.Data.ExteriorWallCase900
         extWalCase900 "Exterior wall"
        annotation (Placement(transformation(extent={{60,60},{74,74}})));

      parameter Buildings.ThermalZones.Detailed.Validation.BESTEST.Data.FloorCase900
        floorCase900 "Floor"
        annotation (Placement(transformation(extent={{80,60},{94,74}})));

      annotation (
    experiment(Tolerance=1e-06, StopTime=3.1536e+07),
    __Dymola_Commands(file="modelica://Buildings/Resources/Scripts/Dymola/ThermalZones/Detailed/Validation/BESTEST/Cases9xx/Case900FF.mos"
            "Simulate and plot"), Documentation(info="<html>
<p>
This model is used for the test case 900FF of the BESTEST validation suite.
Case 900FF is a heavy-weight building.
The room temperature is free floating.
</p>
</html>",     revisions="<html>
<ul>
<li>
July 29, 2016, by Michael Wetter:<br/>
Added missing parameter declarations.
This is for
<a href=\"https://github.com/lbl-srg/modelica-buildings/issues/543\">issue 543</a>.
</li>
<li>
October 6, 2011, by Michael Wetter:<br/>
First implementation.
</li>
</ul>
</html>"),
        Icon(graphics={                     Text(
            extent={{-152,202},{148,162}},
            textString="%name",
            lineColor={0,0,255})}));
    end Case900FF;

    model Case600FF
      "Basic test with light-weight construction and free floating temperature"

      replaceable package MediumA = Buildings.Media.Air(extraPropertiesNames={"CO2"}) "Medium model";
      parameter Modelica.SIunits.MassFlowRate mAir_flow_nominal "Nominal air mass flow rate";
      parameter Modelica.SIunits.Angle S_=
        Buildings.Types.Azimuth.S "Azimuth for south walls";
      parameter Modelica.SIunits.Angle E_=
        Buildings.Types.Azimuth.E "Azimuth for east walls";
      parameter Modelica.SIunits.Angle W_=
        Buildings.Types.Azimuth.W "Azimuth for west walls";
      parameter Modelica.SIunits.Angle N_=
        Buildings.Types.Azimuth.N "Azimuth for north walls";
      parameter Modelica.SIunits.Angle C_=
        Buildings.Types.Tilt.Ceiling "Tilt for ceiling";
      parameter Modelica.SIunits.Angle F_=
        Buildings.Types.Tilt.Floor "Tilt for floor";
      parameter Modelica.SIunits.Angle Z_=
        Buildings.Types.Tilt.Wall "Tilt for wall";
      parameter Integer nConExtWin = 1 "Number of constructions with a window";
      parameter Integer nConBou = 1
        "Number of surface that are connected to constructions that are modeled inside the room";
      parameter Buildings.HeatTransfer.Data.OpaqueConstructions.Generic matExtWal(
        nLay=3,
        absIR_a=0.9,
        absIR_b=0.9,
        absSol_a=0.6,
        absSol_b=0.6,
        material={Buildings.HeatTransfer.Data.Solids.Generic(
            x=0.009,
            k=0.140,
            c=900,
            d=530,
            nStaRef=Buildings.ThermalZones.Detailed.Validation.BESTEST.nStaRef),
                             Buildings.HeatTransfer.Data.Solids.Generic(
            x=0.066,
            k=0.040,
            c=840,
            d=12,
            nStaRef=Buildings.ThermalZones.Detailed.Validation.BESTEST.nStaRef),
                             Buildings.HeatTransfer.Data.Solids.Generic(
            x=0.012,
            k=0.160,
            c=840,
            d=950,
            nStaRef=Buildings.ThermalZones.Detailed.Validation.BESTEST.nStaRef)})
                               "Exterior wall"
        annotation (Placement(transformation(extent={{20,84},{34,98}})));
      parameter Buildings.HeatTransfer.Data.OpaqueConstructions.Generic
                                                              matFlo(final nLay=
               2,
        absIR_a=0.9,
        absIR_b=0.9,
        absSol_a=0.6,
        absSol_b=0.6,
        material={Buildings.HeatTransfer.Data.Solids.Generic(
            x=1.003,
            k=0.040,
            c=0,
            d=0,
            nStaRef=Buildings.ThermalZones.Detailed.Validation.BESTEST.nStaRef),
                             Buildings.HeatTransfer.Data.Solids.Generic(
            x=0.025,
            k=0.140,
            c=1200,
            d=650,
            nStaRef=Buildings.ThermalZones.Detailed.Validation.BESTEST.nStaRef)})
                               "Floor"
        annotation (Placement(transformation(extent={{80,84},{94,98}})));
       parameter Buildings.HeatTransfer.Data.Solids.Generic soil(
        x=2,
        k=1.3,
        c=800,
        d=1500) "Soil properties"
        annotation (Placement(transformation(extent={{40,40},{60,60}})));

      Buildings.ThermalZones.Detailed.MixedAir roo(
        redeclare package Medium = MediumA,
        hRoo=2.7,
        nConExtWin=nConExtWin,
        nConBou=1,
        use_C_flow=true,
        C_start={400e-6},
        nPorts=5,
        energyDynamics=Modelica.Fluid.Types.Dynamics.FixedInitial,
        AFlo=48,
        datConBou(
          layers={matFlo},
          each A=48,
          each til=F_),
        datConExt(
          layers={roof,matExtWal,matExtWal,matExtWal},
          A={48,6*2.7,6*2.7,8*2.7},
          til={C_,Z_,Z_,Z_},
          azi={S_,W_,E_,N_}),
        nConExt=4,
        nConPar=0,
        nSurBou=0,
        datConExtWin(
          layers={matExtWal},
          A={8*2.7},
          glaSys={window600},
          wWin={2*3},
          hWin={2},
          fFra={0.001},
          til={Z_},
          azi={S_}),
        lat=weaDat.lat,
        massDynamics=Modelica.Fluid.Types.Dynamics.SteadyState)
        "Room model for Case 600"
        annotation (Placement(transformation(extent={{36,-30},{66,0}})));
      Modelica.Blocks.Routing.Multiplex3 multiplex3_1
        annotation (Placement(transformation(extent={{-18,64},{-10,72}})));
      Buildings.BoundaryConditions.WeatherData.ReaderTMY3 weaDat(filNam=
            Modelica.Utilities.Files.loadResource("modelica://Buildings/Resources/weatherdata/DRYCOLD.mos"),
          computeWetBulbTemperature=true)
        annotation (Placement(transformation(extent={{98,-94},{86,-82}})));
      Modelica.Blocks.Sources.Constant uSha(k=0)
        "Control signal for the shading device"
        annotation (Placement(transformation(extent={{-28,76},{-20,84}})));
      Modelica.Blocks.Routing.Replicator replicator(nout=max(1,nConExtWin))
        annotation (Placement(transformation(extent={{-12,76},{-4,84}})));
      Modelica.Thermal.HeatTransfer.Sources.FixedTemperature TSoi[nConBou](each T=
            283.15) "Boundary condition for construction"
                                              annotation (Placement(transformation(
            extent={{0,0},{-8,8}},
            origin={72,-52})));
      parameter Buildings.HeatTransfer.Data.OpaqueConstructions.Generic roof(nLay=3,
        absIR_a=0.9,
        absIR_b=0.9,
        absSol_a=0.6,
        absSol_b=0.6,
        material={Buildings.HeatTransfer.Data.Solids.Generic(
            x=0.019,
            k=0.140,
            c=900,
            d=530,
            nStaRef=Buildings.ThermalZones.Detailed.Validation.BESTEST.nStaRef),
                             Buildings.HeatTransfer.Data.Solids.Generic(
            x=0.1118,
            k=0.040,
            c=840,
            d=12,
            nStaRef=Buildings.ThermalZones.Detailed.Validation.BESTEST.nStaRef),
                             Buildings.HeatTransfer.Data.Solids.Generic(
            x=0.010,
            k=0.160,
            c=840,
            d=950,
            nStaRef=Buildings.ThermalZones.Detailed.Validation.BESTEST.nStaRef)})
                               "Roof"
        annotation (Placement(transformation(extent={{60,84},{74,98}})));
      Buildings.ThermalZones.Detailed.Validation.BESTEST.Data.Win600
             window600(
        UFra=3,
        haveExteriorShade=false,
        haveInteriorShade=false) "Window"
        annotation (Placement(transformation(extent={{40,84},{54,98}})));
      Buildings.HeatTransfer.Conduction.SingleLayer soi(
        A=48,
        material=soil,
        steadyStateInitial=true,
        stateAtSurface_a=false,
        stateAtSurface_b=true,
        T_a_start=283.15,
        T_b_start=283.75) "2m deep soil (per definition on p.4 of ASHRAE 140-2007)"
        annotation (Placement(transformation(
            extent={{5,-5},{-3,3}},
            rotation=-90,
            origin={57,-35})));
      Buildings.Fluid.Sources.MassFlowSource_WeatherData
                                               sinInf(
        redeclare package Medium = MediumA,
        m_flow=1,
        use_m_flow_in=true,
        use_C_in=true,
        nPorts=1) "Sink model for air infiltration"
        annotation (Placement(transformation(extent={{-10,-66},{2,-54}})));
      Modelica.Blocks.Sources.Constant InfiltrationRate(k=-48*2.7*0.5/3600)
        "0.41 ACH adjusted for the altitude (0.5 at sea level)"
        annotation (Placement(transformation(extent={{-96,-78},{-88,-70}})));
      Modelica.Blocks.Math.Product product
        annotation (Placement(transformation(extent={{-50,-60},{-40,-50}})));
      Buildings.Fluid.Sensors.Density density(redeclare package Medium = MediumA)
        "Air density inside the building"
        annotation (Placement(transformation(extent={{-40,-76},{-50,-66}})));
      Modelica.Thermal.HeatTransfer.Sensors.TemperatureSensor TRooAirSen
        "Room air temperature"
        annotation (Placement(transformation(extent={{80,16},{90,26}})));
      replaceable parameter
        Buildings.ThermalZones.Detailed.Validation.BESTEST.Data.StandardResultsFreeFloating
          staRes(
            minT( Min=-18.8+273.15, Max=-15.6+273.15, Mean=-17.6+273.15),
            maxT( Min=64.9+273.15,  Max=69.5+273.15,  Mean=66.2+273.15),
            meanT(Min=24.2+273.15,  Max=25.9+273.15,  Mean=25.1+273.15))
              constrainedby Modelica.Icons.Record
        "Reference results from ASHRAE/ANSI Standard 140"
        annotation (Placement(transformation(extent={{80,40},{94,54}})));
      Modelica.Blocks.Math.MultiSum multiSum(nu=1)
        annotation (Placement(transformation(extent={{-78,-80},{-66,-68}})));

      Modelica.Blocks.Interfaces.RealOutput TRooAir "Room air temperature"
        annotation (Placement(transformation(extent={{160,-10},{180,10}}),
            iconTransformation(extent={{160,-10},{180,10}})));
      Buildings.Fluid.Sources.MassFlowSource_WeatherData
                                               souInf(
        redeclare package Medium = MediumA,
        m_flow=1,
        use_m_flow_in=true,
        use_C_in=true,
        nPorts=1) "source model for air infiltration"
        annotation (Placement(transformation(extent={{4,-46},{16,-34}})));
      Modelica.Blocks.Math.Gain gain(k=-1)
        annotation (Placement(transformation(extent={{-18,-40},{-8,-30}})));
      Modelica.Fluid.Interfaces.FluidPort_a supplyAir(redeclare final package
          Medium = MediumA) "Supply air"
        annotation (Placement(transformation(extent={{-110,10},{-90,30}}),
            iconTransformation(extent={{-110,10},{-90,30}})));
      Modelica.Fluid.Interfaces.FluidPort_b returnAir(redeclare final package
          Medium = MediumA) "Return air"
        annotation (Placement(transformation(extent={{-110,-30},{-90,-10}}),
            iconTransformation(extent={{-110,-30},{-90,-10}})));
      InternalLoad lig(
        radFraction=0.5,
        latPower_nominal=0,
        senPower_nominal=11.8)
        annotation (Placement(transformation(extent={{-100,40},{-80,60}})));
      InternalLoad equ(
        latPower_nominal=0,
        senPower_nominal=5.4,
        radFraction=0.7)
        annotation (Placement(transformation(extent={{-100,60},{-80,80}})));
      OccupancyLoad occ(
        radFraction=0.6,
        co2Gen=8.64e-6,
        occ_density=2/48,
        senPower=73,
        latPower=45)
        annotation (Placement(transformation(extent={{-100,80},{-80,100}})));
      Modelica.Blocks.Math.MultiSum sumRad(nu=3) "Sum of radiant internal gains"
        annotation (Placement(transformation(extent={{-52,82},{-40,94}})));
      Modelica.Blocks.Math.MultiSum sumCon(nu=3) "Sum of convective internal gains"
        annotation (Placement(transformation(extent={{-52,62},{-40,74}})));
      Modelica.Blocks.Math.MultiSum sumLat(nu=3) "Sum of latent internal gains"
        annotation (Placement(transformation(extent={{-52,42},{-40,54}})));
      Buildings.Utilities.IO.SignalExchange.Read reaPLig(y(unit="W"), description=
            "Lighting power submeter") "Read lighting power consumption"
        annotation (Placement(transformation(extent={{-20,20},{0,40}})));
      Buildings.Utilities.IO.SignalExchange.Read reaPPlu(y(unit="W"), description=
            "Plug load power submeter") "Read plug load power consumption"
        annotation (Placement(transformation(extent={{-20,0},{0,20}})));
      Modelica.Blocks.Math.MultiSum sumLig(k=fill(roo.AFlo, 2), nu=2)
        "Lighting power consumption"
        annotation (Placement(transformation(extent={{-52,24},{-40,36}})));
      Modelica.Blocks.Math.MultiSum sumPlu(k=fill(roo.AFlo, 2), nu=2)
        "Plug power consumption"
        annotation (Placement(transformation(extent={{-52,4},{-40,16}})));
      Buildings.Utilities.IO.SignalExchange.Read reaTRooAir(
        description="Zone air temperature",
        KPIs=Buildings.Utilities.IO.SignalExchange.SignalTypes.SignalsForKPIs.AirZoneTemperature,
        y(unit="K")) "Read room air temperature"
        annotation (Placement(transformation(extent={{120,-10},{140,10}})));

      Buildings.Utilities.IO.SignalExchange.Read reaCO2RooAir(
        description="Zone air CO2 concentration",
        KPIs=Buildings.Utilities.IO.SignalExchange.SignalTypes.SignalsForKPIs.CO2Concentration,
        y(unit="ppm"))
                     "Read room air CO2 concentration"
        annotation (Placement(transformation(extent={{130,-40},{150,-20}})));

      Modelica.Blocks.Interfaces.RealOutput CO2RooAir(unit="ppm") "Room air CO2 concentration"
        annotation (Placement(transformation(extent={{160,-50},{180,-30}}),
            iconTransformation(extent={{160,-50},{180,-30}})));
      Modelica.Blocks.Sources.Constant conCO2Out(k=400e-6*Modelica.Media.IdealGases.Common.SingleGasesData.CO2.MM
            /Modelica.Media.IdealGases.Common.SingleGasesData.Air.MM)
        "Outside air CO2 concentration"
        annotation (Placement(transformation(extent={{-34,-72},{-26,-64}})));
      Buildings.Fluid.Sensors.TraceSubstancesTwoPort senCO2(
        redeclare package Medium = MediumA,
        m_flow_nominal=mAir_flow_nominal)
                        "CO2 sensor"
        annotation (Placement(transformation(extent={{28,-70},{8,-50}})));
      Modelica.Blocks.Math.Gain gaiCO2Gen(k=roo.AFlo)
        "Gain for CO2 generation by floor area"
        annotation (Placement(transformation(extent={{-50,-18},{-40,-8}})));
      Modelica.Blocks.Math.Gain gaiPPM(k=1e6) "Convert mass fraction to PPM"
        annotation (Placement(transformation(extent={{100,-40},{120,-20}})));
      Buildings.Fluid.Sensors.Conversions.To_VolumeFraction conMasVolFra(MMMea=
            Modelica.Media.IdealGases.Common.SingleGasesData.CO2.MM)
        "Conversion from mass fraction CO2 to volume fraction CO2"
        annotation (Placement(transformation(extent={{70,-40},{90,-20}})));
      Buildings.Utilities.IO.SignalExchange.WeatherStation weaSta
        "BOPTEST weather station"
        annotation (Placement(transformation(extent={{60,-80},{40,-60}})));
    equation
      connect(multiplex3_1.y, roo.qGai_flow) annotation (Line(
          points={{-9.6,68},{20,68},{20,-9},{34.8,-9}},
          color={0,0,127},
          smooth=Smooth.None));
      connect(roo.uSha, replicator.y) annotation (Line(
          points={{34.8,-1.5},{24,-1.5},{24,80},{-3.6,80}},
          color={0,0,127},
          smooth=Smooth.None));
      connect(weaDat.weaBus, roo.weaBus)  annotation (Line(
          points={{86,-88},{80.07,-88},{80.07,-1.575},{64.425,-1.575}},
          color={255,204,51},
          thickness=0.5,
          smooth=Smooth.None));
      connect(uSha.y, replicator.u) annotation (Line(
          points={{-19.6,80},{-12.8,80}},
          color={0,0,127},
          smooth=Smooth.None));
      connect(product.y, sinInf.m_flow_in)       annotation (Line(
          points={{-39.5,-55},{-36,-55},{-36,-55.2},{-10,-55.2}},
          color={0,0,127},
          smooth=Smooth.None));
      connect(density.port, roo.ports[1])  annotation (Line(
          points={{-45,-76},{32,-76},{32,-24.9},{39.75,-24.9}},
          color={0,127,255},
          smooth=Smooth.None));
      connect(density.d, product.u2) annotation (Line(
          points={{-50.5,-71},{-56,-71},{-56,-58},{-51,-58}},
          color={0,0,127},
          smooth=Smooth.None));
      connect(TSoi[1].port, soi.port_a) annotation (Line(
          points={{64,-48},{56,-48},{56,-40}},
          color={191,0,0},
          smooth=Smooth.None));
      connect(soi.port_b, roo.surf_conBou[1]) annotation (Line(
          points={{56,-32},{56,-27},{55.5,-27}},
          color={191,0,0},
          smooth=Smooth.None));
      connect(multiSum.y, product.u1) annotation (Line(
          points={{-64.98,-74},{-54,-74},{-54,-52},{-51,-52}},
          color={0,0,127},
          smooth=Smooth.None));
      connect(InfiltrationRate.y, multiSum.u[1]) annotation (Line(
          points={{-87.6,-74},{-78,-74}},
          color={0,0,127},
          smooth=Smooth.None));

      connect(TRooAirSen.port, roo.heaPorAir) annotation (Line(points={{80,21},{60,
              21},{60,-15},{50.25,-15}}, color={191,0,0}));
      connect(gain.y, souInf.m_flow_in) annotation (Line(points={{-7.5,-35},{-4.75,-35},
              {-4.75,-35.2},{4,-35.2}},        color={0,0,127}));
      connect(gain.u, sinInf.m_flow_in) annotation (Line(points={{-19,-35},{-30,-35},
              {-30,-55.2},{-10,-55.2}}, color={0,0,127}));
      connect(souInf.ports[1], roo.ports[2]) annotation (Line(points={{16,-40},{20,-40},
              {20,-23.7},{39.75,-23.7}},      color={0,127,255}));
      connect(supplyAir, roo.ports[3]) annotation (Line(points={{-100,20},{-80,20},{
              -80,-22.5},{39.75,-22.5}},
                                       color={0,127,255}));
      connect(occ.rad, sumRad.u[1]) annotation (Line(points={{-79,94},{-60,94},{-60,
              90.8},{-52,90.8}}, color={0,0,127}));
      connect(equ.rad, sumRad.u[2]) annotation (Line(points={{-79,74},{-60,74},{-60,
              88},{-52,88}}, color={0,0,127}));
      connect(lig.rad, sumRad.u[3]) annotation (Line(points={{-79,54},{-58,54},{-58,
              85.2},{-52,85.2}}, color={0,0,127}));
      connect(occ.con, sumCon.u[1]) annotation (Line(points={{-79,90},{-64,90},{-64,
              70.8},{-52,70.8}}, color={0,0,127}));
      connect(equ.con, sumCon.u[2]) annotation (Line(points={{-79,70},{-68,70},{-68,
              68},{-52,68}}, color={0,0,127}));
      connect(lig.con, sumCon.u[3]) annotation (Line(points={{-79,50},{-60,50},{-60,
              65.2},{-52,65.2}}, color={0,0,127}));
      connect(occ.lat, sumLat.u[1]) annotation (Line(points={{-79,86},{-72,86},{-72,
              50.8},{-52,50.8}}, color={0,0,127}));
      connect(equ.lat, sumLat.u[2]) annotation (Line(points={{-79,66},{-74,66},{-74,
              48},{-52,48}}, color={0,0,127}));
      connect(lig.lat, sumLat.u[3]) annotation (Line(points={{-79,46},{-76,46},{-76,
              44},{-52,44},{-52,45.2}}, color={0,0,127}));
      connect(sumRad.y, multiplex3_1.u1[1]) annotation (Line(points={{-38.98,88},{
              -32,88},{-32,70.8},{-18.8,70.8}}, color={0,0,127}));
      connect(sumCon.y, multiplex3_1.u2[1])
        annotation (Line(points={{-38.98,68},{-18.8,68}}, color={0,0,127}));
      connect(sumLat.y, multiplex3_1.u3[1]) annotation (Line(points={{-38.98,48},{
              -32,48},{-32,65.2},{-18.8,65.2}}, color={0,0,127}));
      connect(sumLig.y, reaPLig.u)
        annotation (Line(points={{-38.98,30},{-22,30}}, color={0,0,127}));
      connect(sumPlu.y, reaPPlu.u)
        annotation (Line(points={{-38.98,10},{-22,10}}, color={0,0,127}));
      connect(equ.con, sumPlu.u[1]) annotation (Line(points={{-79,70},{-68,70},{-68,
              12.1},{-52,12.1}}, color={0,0,127}));
      connect(equ.rad, sumPlu.u[2]) annotation (Line(points={{-79,74},{-70,74},{-70,
              7.9},{-52,7.9}}, color={0,0,127}));
      connect(lig.rad, sumLig.u[1]) annotation (Line(points={{-79,54},{-58,54},{-58,
              32.1},{-52,32.1}}, color={0,0,127}));
      connect(lig.con, sumLig.u[2]) annotation (Line(points={{-79,50},{-60,50},{-60,
              27.9},{-52,27.9}}, color={0,0,127}));
      connect(TRooAirSen.T, reaTRooAir.u) annotation (Line(points={{90,21},{96,21},
              {96,0},{118,0}}, color={0,0,127}));
      connect(reaTRooAir.y, TRooAir)
        annotation (Line(points={{141,0},{170,0}}, color={0,0,127}));
      connect(reaCO2RooAir.y, CO2RooAir) annotation (Line(points={{151,-30},{156,
              -30},{156,-40},{170,-40}},
                                    color={0,0,127}));
      connect(returnAir, roo.ports[4]) annotation (Line(points={{-100,-20},{-30,-20},
              {-30,-21.3},{39.75,-21.3}}, color={0,127,255}));
      connect(sinInf.ports[1], senCO2.port_b)
        annotation (Line(points={{2,-60},{8,-60}}, color={0,127,255}));
      connect(senCO2.port_a, roo.ports[5]) annotation (Line(points={{28,-60},{30,-60},
              {30,-20.1},{39.75,-20.1}}, color={0,127,255}));
      connect(occ.co2, gaiCO2Gen.u) annotation (Line(points={{-79,82},{-66,82},{-66,
              -13},{-51,-13}}, color={0,0,127}));
      connect(gaiCO2Gen.y, roo.C_flow[1]) annotation (Line(points={{-39.5,-13},{-1.75,
              -13},{-1.75,-12.9},{34.8,-12.9}}, color={0,0,127}));
      connect(weaDat.weaBus, sinInf.weaBus) annotation (Line(
          points={{86,-88},{-20,-88},{-20,-59.88},{-10,-59.88}},
          color={255,204,51},
          thickness=0.5));
      connect(weaDat.weaBus, souInf.weaBus) annotation (Line(
          points={{86,-88},{-20,-88},{-20,-39.88},{4,-39.88}},
          color={255,204,51},
          thickness=0.5));
      connect(conCO2Out.y, sinInf.C_in[1]) annotation (Line(points={{-25.6,-68},{-16,
              -68},{-16,-64.8},{-10,-64.8}}, color={0,0,127}));
      connect(conCO2Out.y, souInf.C_in[1]) annotation (Line(points={{-25.6,-68},{-16,
              -68},{-16,-44.8},{4,-44.8}}, color={0,0,127}));
      connect(gaiPPM.y, reaCO2RooAir.u)
        annotation (Line(points={{121,-30},{128,-30}}, color={0,0,127}));
      connect(senCO2.C, conMasVolFra.m)
        annotation (Line(points={{18,-49},{18,-30},{69,-30}}, color={0,0,127}));
      connect(conMasVolFra.V, gaiPPM.u)
        annotation (Line(points={{91,-30},{98,-30}}, color={0,0,127}));
      connect(weaSta.weaBus, roo.weaBus) annotation (Line(
          points={{59.9,-70.1},{80,-70},{80.07,-70},{80.07,-1.575},{64.425,-1.575}},
          color={255,204,51},
          thickness=0.5));

      annotation (
    experiment(Tolerance=1e-06, StopTime=3.1536e+07),
    __Dymola_Commands(file="modelica://Buildings/Resources/Scripts/Dymola/ThermalZones/Detailed/Validation/BESTEST/Cases6xx/Case600FF.mos"
            "Simulate and plot"), Documentation(info="<html>
<p>
This model is used for the test case 600FF of the BESTEST validation suite.
Case 600FF is a light-weight building.
The room temperature is free floating.
</p>
</html>",     revisions="<html>
<ul>
<li>
October 29, 2016, by Michael Wetter:<br/>
Placed a capacity at the room-facing surface
to reduce the dimension of the nonlinear system of equations,
which generally decreases computing time.<br/>
Removed the pressure drop element which is not needed.<br/>
Linearized the radiative heat transfer, which is the default in
the library, and avoids a large nonlinear system of equations.<br/>
This is for
<a href=\"https://github.com/lbl-srg/modelica-buildings/issues/565\">issue 565</a>.
</li>
<li>
December 22, 2014 by Michael Wetter:<br/>
Removed <code>Modelica.Fluid.System</code>
to address issue
<a href=\"https://github.com/lbl-srg/modelica-buildings/issues/311\">#311</a>.
</li>
<li>
October 9, 2013, by Michael Wetter:<br/>
Implemented soil properties using a record so that <code>TSol</code> and
<code>TLiq</code> are assigned.
This avoids an error when the model is checked in the pedantic mode.
</li>
<li>
July 15, 2012, by Michael Wetter:<br/>
Added reference results.
Changed implementation to make this model the base class
for all BESTEST cases.
Added computation of hourly and annual averaged room air temperature.
<li>
October 6, 2011, by Michael Wetter:<br/>
First implementation.
</li>
</ul>
</html>"),
        Icon(graphics={
            Rectangle(
              extent={{-160,-160},{160,160}},
              lineColor={95,95,95},
              fillColor={95,95,95},
              fillPattern=FillPattern.Solid),
            Rectangle(
              extent={{-140,138},{140,-140}},
              pattern=LinePattern.None,
              lineColor={117,148,176},
              fillColor={170,213,255},
              fillPattern=FillPattern.Sphere),
            Rectangle(
              extent={{140,70},{160,-70}},
              lineColor={95,95,95},
              fillColor={255,255,255},
              fillPattern=FillPattern.Solid),
            Rectangle(
              extent={{146,70},{154,-70}},
              lineColor={95,95,95},
              fillColor={170,213,255},
              fillPattern=FillPattern.Solid)}));
    end Case600FF;

    model FanCoilUnit "Four-pipe fan coil unit model"
      replaceable package Medium1 = Buildings.Media.Air(extraPropertiesNames={"CO2"});
      replaceable package Medium2 = Buildings.Media.Water;
      parameter Modelica.SIunits.MassFlowRate mAir_flow_nominal=0.55 "Nominal air flowrate" annotation (Dialog(group="Air"));
      parameter Modelica.SIunits.DimensionlessRatio minSpe=0.2 "Minimum fan speed" annotation (Dialog(group="Air"));
      parameter Modelica.SIunits.Power QCooCap=3666 "Cooling coil capacity" annotation (Dialog(group="Coils"));
      parameter Modelica.SIunits.Power QHeaCap=7000 "Heating coil capacity" annotation (Dialog(group="Coils"));
      parameter Modelica.SIunits.DimensionlessRatio COP = 3 "Assumed COP of chiller supplying chilled water to FCU in [W_thermal/W_electric]" annotation (Dialog(group="Plant"));
      parameter Modelica.SIunits.DimensionlessRatio eff = 0.9 "Assumed efficiency of gas boiler supplying hot water to FCU in [W_gas/W_thermal]" annotation (Dialog(group="Plant"));
      final parameter Modelica.SIunits.Pressure dpAir_nominal=185 "Nominal supply air pressure";
      final parameter Modelica.SIunits.MassFlowRate mCoo_flow_nominal=QCooCap/(4200*5) "Nominal chilled water flowrate";
      final parameter Modelica.SIunits.MassFlowRate mHea_flow_nominal=QHeaCap/(4200*20) "Nominal heating water flowrate";
      final parameter Modelica.SIunits.Pressure dpCoo_nominal=((mCoo_flow_nominal/1000)*3600/(0.865*1))^2*1e5 "Nominal chilled water pressure drop";
      final parameter Modelica.SIunits.Pressure dpHea_nominal=((mHea_flow_nominal/1000)*3600/(0.865*1))^2*1e5 "Nominal heating water pressure drop";
      Modelica.Fluid.Interfaces.FluidPort_a returnAir(redeclare final package
          Medium = Medium1) "Return air" annotation (Placement(transformation(
              extent={{130,-170},{150,-150}}),
                                            iconTransformation(extent={{130,-170},{
                150,-150}})));
      Modelica.Fluid.Interfaces.FluidPort_b supplyAir(redeclare final package
          Medium = Medium1) "Supply air"
        annotation (Placement(transformation(extent={{130,90},{150,110}}),
            iconTransformation(extent={{130,90},{150,110}})));
      Buildings.Fluid.Movers.SpeedControlled_y     fan(redeclare package Medium
          = Medium1, per(
          pressure(V_flow={0,mAir_flow_nominal/1.2}, dp={dpAir_nominal,0}),
          use_powerCharacteristic=true,
          power(V_flow={0,mAir_flow_nominal/1.2}, P={0,dpAir_nominal/0.7/0.7*
                mAir_flow_nominal/1.2})))              annotation (Placement(
            transformation(
            extent={{-10,-10},{10,10}},
            rotation=90,
            origin={0,-120})));

      Modelica.Blocks.Interfaces.RealInput uCooVal
        "Control signal for cooling valve"
        annotation (Placement(transformation(extent={{-180,80},{-140,120}})));
      Modelica.Blocks.Interfaces.RealInput uFan "Fan speed signal"
        annotation (Placement(transformation(extent={{-180,-120},{-140,-80}})));
      Buildings.Fluid.Sensors.TemperatureTwoPort senSupTem(redeclare package
          Medium =
            Medium1, m_flow_nominal=mAir_flow_nominal)
        annotation (Placement(transformation(extent={{90,90},{110,110}})));
      Buildings.Fluid.Sensors.TemperatureTwoPort senRetTem(redeclare package
          Medium =
            Medium1, m_flow_nominal=mAir_flow_nominal)
        annotation (Placement(transformation(extent={{110,-170},{90,-150}})));
      Buildings.Fluid.Sensors.MassFlowRate senSupFlo(redeclare package Medium
          = Medium1)
        annotation (Placement(transformation(extent={{20,90},{40,110}})));
      Modelica.Blocks.Interfaces.RealOutput PFan "Fan electrical power consumption"
        annotation (Placement(transformation(extent={{140,130},{160,150}})));
      Modelica.Blocks.Interfaces.RealOutput PHea "Heating power"
        annotation (Placement(transformation(extent={{140,150},{160,170}})));
      Modelica.Blocks.Interfaces.RealOutput PCoo "Cooling power"
        annotation (Placement(transformation(extent={{140,170},{160,190}})));
      Modelica.Blocks.Math.Gain powHea(k=eff)
        annotation (Placement(transformation(extent={{-8,150},{12,170}})));
      Modelica.Blocks.Math.Gain powCoo(k=1/COP)
        annotation (Placement(transformation(extent={{-8,170},{12,190}})));
      Buildings.Utilities.IO.SignalExchange.Read reaTSup(description=
            "Supply air temperature", y(unit="K")) "Read supply air temperature"
        annotation (Placement(transformation(extent={{110,110},{130,130}})));
      Buildings.Utilities.IO.SignalExchange.Read reaTRet(y(unit="K"), description=
            "Return air temperature") "Read return air temperature"
        annotation (Placement(transformation(extent={{110,-150},{130,-130}})));
      Buildings.Utilities.IO.SignalExchange.Read reaFloSup(y(unit="kg/s"), description=
            "Supply air mass flow rate") "Read supply air flowrate"
        annotation (Placement(transformation(extent={{40,110},{60,130}})));
      Buildings.Utilities.IO.SignalExchange.Read reaFanSpeSet(y(unit="1"), description=
            "Supply fan speed setpoint") "Read supply fan speed setpoint"
        annotation (Placement(transformation(extent={{20,-120},{40,-100}})));
      Buildings.Fluid.Actuators.Valves.TwoWayEqualPercentage cooVal(
        redeclare package Medium = Medium2,
        m_flow_nominal=mCoo_flow_nominal,
        dpValve_nominal(displayUnit="bar") = dpCoo_nominal)
        annotation (Placement(transformation(extent={{50,10},{70,30}})));
      Buildings.Fluid.Sensors.TemperatureTwoPort senCooSupTem(redeclare package
          Medium = Medium2, m_flow_nominal=mCoo_flow_nominal)
        annotation (Placement(transformation(extent={{100,20},{80,40}})));
      Buildings.Fluid.Sensors.TemperatureTwoPort senCooRetTem(redeclare package
          Medium = Medium2, m_flow_nominal=mCoo_flow_nominal)
        annotation (Placement(transformation(extent={{78,-28},{98,-8}})));
      Buildings.Fluid.Sensors.MassFlowRate senCooMasFlo(redeclare package
          Medium =
            Medium2)
        annotation (Placement(transformation(extent={{40,10},{20,30}})));
      Buildings.Fluid.Actuators.Valves.TwoWayEqualPercentage heaVal(
        redeclare package Medium = Medium2,
        m_flow_nominal=mHea_flow_nominal,
        dpValve_nominal(displayUnit="bar") = dpHea_nominal)
        annotation (Placement(transformation(extent={{50,-80},{70,-60}})));
      Buildings.Fluid.Sensors.TemperatureTwoPort senHeaSupTem(redeclare package
          Medium = Medium2, m_flow_nominal=mHea_flow_nominal)
        annotation (Placement(transformation(extent={{100,-70},{80,-50}})));
      Buildings.Fluid.Sensors.TemperatureTwoPort senHeaRetTem(redeclare package
          Medium = Medium2, m_flow_nominal=mHea_flow_nominal)
        annotation (Placement(transformation(extent={{80,-120},{100,-100}})));
      Buildings.Fluid.Sensors.MassFlowRate senHeaMasFlo(redeclare package
          Medium =
            Medium2)
        annotation (Placement(transformation(extent={{40,-80},{20,-60}})));
      Buildings.Fluid.Sources.Boundary_pT souCoo(
        redeclare package Medium = Medium2,
        p=101325 + dpCoo_nominal,
        T=280.35,                                nPorts=1)
        "Source for chilled water"
        annotation (Placement(transformation(extent={{140,20},{120,40}})));
      Buildings.Fluid.Sources.Boundary_pT sinCoo(redeclare package Medium = Medium2,
        p(displayUnit="Pa") = 101325,
        nPorts=1)                                          "Sink for chilled water"
        annotation (Placement(transformation(extent={{140,-28},{120,-8}})));
      Buildings.Fluid.Sources.Boundary_pT sinHea(redeclare package Medium = Medium2,
        p(displayUnit="Pa") = 101325,            nPorts=1) "Sink for heating water"
        annotation (Placement(transformation(extent={{140,-120},{120,-100}})));
      Buildings.Fluid.Sources.Boundary_pT souHea(
        redeclare package Medium = Medium2,
        p=101325 + dpHea_nominal,
        T=333.15,                                nPorts=1)
        "Source for heating water"
        annotation (Placement(transformation(extent={{140,-70},{120,-50}})));
      Modelica.Blocks.Interfaces.RealInput uHeaVal
        "Control signal for heating valve"
        annotation (Placement(transformation(extent={{-180,-20},{-140,20}})));
      Buildings.Fluid.HeatExchangers.ConstantEffectiveness cooCoi(
        redeclare package Medium1 = Medium1,
        redeclare package Medium2 = Medium2,
        m1_flow_nominal=mAir_flow_nominal,
        m2_flow_nominal=mCoo_flow_nominal,
        dp1_nominal=dpAir_nominal/2,
        dp2_nominal=0) "Cooling coil" annotation (Placement(transformation(
            extent={{-10,-10},{10,10}},
            rotation=90,
            origin={0,10})));
      Buildings.Fluid.HeatExchangers.ConstantEffectiveness heaCoi(
        redeclare package Medium1 = Medium1,
        redeclare package Medium2 = Medium2,
        m1_flow_nominal=mAir_flow_nominal,
        m2_flow_nominal=mHea_flow_nominal,
        dp1_nominal=dpAir_nominal/2,
        dp2_nominal=0) "Heating coil" annotation (Placement(transformation(
            extent={{-10,-10},{10,10}},
            rotation=90,
            origin={0,-80})));
      Buildings.Utilities.IO.SignalExchange.Read reaHeaVal(y(unit="1"), description="Heating valve control signal")
        "Read heating valve control signal"
        annotation (Placement(transformation(extent={{-80,-10},{-60,10}})));
      Buildings.Utilities.IO.SignalExchange.Read reaCooVal(y(unit="1"), description="Cooling valve control signal")
        "Read cooling valve control signal"
        annotation (Placement(transformation(extent={{-80,90},{-60,110}})));
      Modelica.Blocks.Sources.RealExpression powCooThe(y=senCooMasFlo.m_flow*(
            inStream(cooCoi.port_b2.h_outflow) - inStream(cooCoi.port_a2.h_outflow)))
                                                       "Cooling thermal power"
        annotation (Placement(transformation(extent={{-60,170},{-40,190}})));
      Modelica.Blocks.Sources.RealExpression powHeaThe(y=-senHeaMasFlo.m_flow*(
            inStream(heaCoi.port_b2.h_outflow) - inStream(heaCoi.port_a2.h_outflow)))
                                                       "Heating thermal power"
        annotation (Placement(transformation(extent={{-60,150},{-40,170}})));
      Modelica.Blocks.Interfaces.BooleanInput uFanSta "Status of fan"
        annotation (Placement(transformation(extent={{-180,-180},{-140,-140}})));
      Buildings.Utilities.IO.SignalExchange.Overwrite oveFan(description="Fan speed control signal",
          u(
          min=0,
          max=1,
          unit="1")) "Overwrite for fan speed control signal"
        annotation (Placement(transformation(extent={{-120,-110},{-100,-90}})));
      FanControl fanControl(minSpe=minSpe)
        annotation (Placement(transformation(extent={{-70,-130},{-50,-110}})));
      Buildings.Utilities.IO.SignalExchange.Overwrite oveCooVal(description="Cooling valve control signal",
          u(
          min=0,
          max=1,
          unit="1")) "Overwrite for cooling valve control signal"
        annotation (Placement(transformation(extent={{-120,90},{-100,110}})));
      Buildings.Utilities.IO.SignalExchange.Overwrite oveHeaVal(description="Heating valve control signal",
          u(
          min=0,
          max=1,
          unit="1")) "Overwrite for heating valve control signal"
        annotation (Placement(transformation(extent={{-120,-10},{-100,10}})));
      Modelica.Blocks.Math.BooleanToReal booleanToReal
        annotation (Placement(transformation(extent={{-134,-166},{-122,-154}})));
      Modelica.Blocks.Math.RealToBoolean realToBoolean
        annotation (Placement(transformation(extent={{-92,-136},{-80,-124}})));
      Buildings.Utilities.IO.SignalExchange.Overwrite oveFanSta(description="Fan status control signal",
          u(
          min=0,
          max=1,
          unit="1")) "Overwrite for fan status control signal"
        annotation (Placement(transformation(extent={{-120,-140},{-100,-120}})));
      Buildings.Utilities.IO.SignalExchange.Read reaFloCoo(y(unit="kg/s"), description=
            "Cooling coil water flow rate") "Read cooling coil water flow rate"
        annotation (Placement(transformation(extent={{40,40},{60,60}})));
      Buildings.Utilities.IO.SignalExchange.Read reaFloHea(y(unit="kg/s"), description=
            "Heating coil water flow rate")
        "Read heating coil supply water flow rate"
        annotation (Placement(transformation(extent={{40,-50},{60,-30}})));
      Buildings.Utilities.IO.SignalExchange.Read reaTHeaLea(description=
            "Heating coil water leaving temperature", y(unit="K"))
        "Read heating coil water leaving temperature"
        annotation (Placement(transformation(extent={{100,-100},{120,-80}})));
      Buildings.Utilities.IO.SignalExchange.Read reaTCooLea(description=
            "Cooling coil water leaving temperature", y(unit="K"))
        "Read cooling coil water leaving temperature"
        annotation (Placement(transformation(extent={{100,-10},{120,10}})));
      Buildings.Utilities.IO.SignalExchange.Read reaPCoo(
        y(unit="W"),
        KPIs=Buildings.Utilities.IO.SignalExchange.SignalTypes.SignalsForKPIs.ElectricPower,
        description="Cooling electrical power consumption")
        "Read power for cooling"
        annotation (Placement(transformation(extent={{70,170},{90,190}})));

      Buildings.Utilities.IO.SignalExchange.Read reaPHea(
        y(unit="W"),
        KPIs=Buildings.Utilities.IO.SignalExchange.SignalTypes.SignalsForKPIs.GasPower,
        description="Heating thermal power consumption") "Read power for heating"
        annotation (Placement(transformation(extent={{70,150},{90,170}})));

      Buildings.Utilities.IO.SignalExchange.Read reaPFan(
        y(unit="W"),
        description="Supply fan electrical power consumption",
        KPIs=Buildings.Utilities.IO.SignalExchange.SignalTypes.SignalsForKPIs.ElectricPower)
        "Read power for supply fan"
        annotation (Placement(transformation(extent={{70,130},{90,150}})));
    equation
      connect(senSupTem.port_b, supplyAir)
        annotation (Line(points={{110,100},{140,100}},
                                                    color={0,127,255}));
      connect(returnAir, senRetTem.port_a)
        annotation (Line(points={{140,-160},{110,-160}},
                                                      color={0,127,255}));
      connect(senRetTem.port_b, fan.port_a) annotation (Line(points={{90,-160},{
              -6.66134e-16,-160},{-6.66134e-16,-130}},
                          color={0,127,255}));
      connect(senSupFlo.port_b, senSupTem.port_a)
        annotation (Line(points={{40,100},{90,100}},
                                                   color={0,127,255}));
      connect(senSupTem.T, reaTSup.u)
        annotation (Line(points={{100,111},{100,120},{108,120}},
                                                           color={0,0,127}));
      connect(senRetTem.T, reaTRet.u)
        annotation (Line(points={{100,-149},{100,-140},{108,-140}},
                                                              color={0,0,127}));
      connect(senSupFlo.m_flow, reaFloSup.u)
        annotation (Line(points={{30,111},{30,120},{38,120}},
                                                           color={0,0,127}));
      connect(senCooSupTem.port_b,cooVal. port_b)
        annotation (Line(points={{80,30},{80,20},{70,20}}, color={0,127,255}));
      connect(senCooMasFlo.port_a,cooVal. port_a)
        annotation (Line(points={{40,20},{50,20}}, color={0,127,255}));
      connect(senHeaSupTem.port_b,heaVal. port_b) annotation (Line(points={{80,-60},
              {80,-70},{70,-70}},          color={0,127,255}));
      connect(heaVal.port_a,senHeaMasFlo. port_a)
        annotation (Line(points={{50,-70},{40,-70}}, color={0,127,255}));
      connect(senCooSupTem.port_a,souCoo. ports[1])
        annotation (Line(points={{100,30},{120,30}},
                                                   color={0,127,255}));
      connect(souHea.ports[1], senHeaSupTem.port_a)
        annotation (Line(points={{120,-60},{100,-60}},
                                                     color={0,127,255}));
      connect(sinHea.ports[1], senHeaRetTem.port_b)
        annotation (Line(points={{120,-110},{100,-110}},
                                                     color={0,127,255}));
      connect(senCooMasFlo.port_b, cooCoi.port_a2)
        annotation (Line(points={{20,20},{6,20}}, color={0,127,255}));
      connect(senCooRetTem.port_a, cooCoi.port_b2)
        annotation (Line(points={{78,-18},{78,0},{6,0}}, color={0,127,255}));
      connect(heaCoi.port_a2, senHeaMasFlo.port_b)
        annotation (Line(points={{6,-70},{20,-70}},    color={0,127,255}));
      connect(heaCoi.port_b2, senHeaRetTem.port_a)
        annotation (Line(points={{6,-90},{80,-90},{80,-110}},
                                                      color={0,127,255}));
      connect(senSupFlo.port_a, cooCoi.port_b1)
        annotation (Line(points={{20,100},{-6,100},{-6,20}}, color={0,127,255}));
      connect(cooCoi.port_a1, heaCoi.port_b1)
        annotation (Line(points={{-6,0},{-6,-70}}, color={0,127,255}));
      connect(fan.port_b, heaCoi.port_a1) annotation (Line(points={{4.44089e-16,
              -110},{4.44089e-16,-90},{-6,-90}},
                                    color={0,127,255}));
      connect(reaCooVal.y, cooVal.y) annotation (Line(points={{-59,100},{-40,100},{
              -40,36},{60,36},{60,32}},
                                    color={0,0,127}));
      connect(powCooThe.y, powCoo.u)
        annotation (Line(points={{-39,180},{-10,180}}, color={0,0,127}));
      connect(powHeaThe.y, powHea.u)
        annotation (Line(points={{-39,160},{-10,160}}, color={0,0,127}));
      connect(senCooRetTem.port_b, sinCoo.ports[1])
        annotation (Line(points={{98,-18},{120,-18}},
                                                 color={0,127,255}));
      connect(uFan, oveFan.u)
        annotation (Line(points={{-160,-100},{-122,-100}},
                                                         color={0,0,127}));
      connect(oveFan.y, fanControl.uFan) annotation (Line(points={{-99,-100},{-90,
              -100},{-90,-116},{-72,-116}},
                                    color={0,0,127}));
      connect(reaHeaVal.y, heaVal.y) annotation (Line(points={{-59,0},{-40,0},{-40,
              -54},{60,-54},{60,-58}},
                                  color={0,0,127}));
      connect(uCooVal, oveCooVal.u)
        annotation (Line(points={{-160,100},{-122,100}},
                                                       color={0,0,127}));
      connect(oveCooVal.y, reaCooVal.u)
        annotation (Line(points={{-99,100},{-82,100}},
                                                     color={0,0,127}));
      connect(oveHeaVal.y, reaHeaVal.u)
        annotation (Line(points={{-99,0},{-82,0}}, color={0,0,127}));
      connect(oveHeaVal.u, uHeaVal)
        annotation (Line(points={{-122,0},{-160,0}}, color={0,0,127}));
      connect(uFanSta, booleanToReal.u)
        annotation (Line(points={{-160,-160},{-135.2,-160}}, color={255,0,255}));
      connect(realToBoolean.y, fanControl.uFanSta) annotation (Line(points={{-79.4,
              -130},{-72,-130},{-72,-122}},
                                    color={255,0,255}));
      connect(oveFanSta.y, realToBoolean.u)
        annotation (Line(points={{-99,-130},{-93.2,-130}},
                                                         color={0,0,127}));
      connect(booleanToReal.y, oveFanSta.u) annotation (Line(points={{-121.4,-160},
              {-110,-160},{-110,-146},{-130,-146},{-130,-130},{-122,-130}},
                                                                         color={0,0,
              127}));
      connect(reaFloCoo.u, senCooMasFlo.m_flow)
        annotation (Line(points={{38,50},{30,50},{30,31}}, color={0,0,127}));
      connect(reaFloHea.u, senHeaMasFlo.m_flow)
        annotation (Line(points={{38,-40},{30,-40},{30,-59}}, color={0,0,127}));
      connect(reaTCooLea.u, senCooRetTem.T)
        annotation (Line(points={{98,0},{88,0},{88,-7}}, color={0,0,127}));
      connect(reaTHeaLea.u, senHeaRetTem.T)
        annotation (Line(points={{98,-90},{90,-90},{90,-99}}, color={0,0,127}));
      connect(powCoo.y, reaPCoo.u)
        annotation (Line(points={{13,180},{68,180}}, color={0,0,127}));
      connect(reaPCoo.y, PCoo)
        annotation (Line(points={{91,180},{150,180}}, color={0,0,127}));
      connect(powHea.y, reaPHea.u)
        annotation (Line(points={{13,160},{68,160}}, color={0,0,127}));
      connect(reaPHea.y, PHea)
        annotation (Line(points={{91,160},{150,160}}, color={0,0,127}));
      connect(fan.P, reaPFan.u) annotation (Line(points={{-9,-109},{-9,-68},{-10,
              -68},{-10,-64},{-16,-64},{-16,140},{68,140}}, color={0,0,127}));
      connect(reaPFan.y, PFan)
        annotation (Line(points={{91,140},{150,140}}, color={0,0,127}));
      connect(fanControl.yFan, fan.y)
        annotation (Line(points={{-49,-120},{-12,-120}}, color={0,0,127}));
      connect(fan.y_actual, reaFanSpeSet.u) annotation (Line(points={{-7,-109},{-7,
              -104},{12,-104},{12,-110},{18,-110}}, color={0,0,127}));
      annotation (Icon(coordinateSystem(preserveAspectRatio=false, extent={{-140,
                -180},{140,180}}),                                  graphics={
                                            Text(
            extent={{-150,184},{150,144}},
            textString="%name",
            lineColor={0,0,255}), Rectangle(
              extent={{-140,180},{140,-180}},
              lineColor={0,0,0},
              fillColor={215,215,215},
              fillPattern=FillPattern.Solid),
                                            Text(
            extent={{-150,238},{150,198}},
            textString="%name",
            lineColor={0,0,255})}),                                  Diagram(
            coordinateSystem(preserveAspectRatio=false, extent={{-140,-180},{140,
                180}})),
        experiment(
          StartTime=20736000,
          StopTime=21600000,
          Interval=599.999616,
          Tolerance=1e-06,
          __Dymola_Algorithm="Cvode"));
    end FanCoilUnit;

    model Thermostat
      "Implements basic control of FCU to maintain zone air temperature"
      parameter Modelica.SIunits.Time occSta = 8*3600 "Occupancy start time" annotation (Dialog(group="Schedule"));
      parameter Modelica.SIunits.Time occEnd = 18*3600 "Occupancy end time" annotation (Dialog(group="Schedule"));
      parameter Modelica.SIunits.DimensionlessRatio minSpe = 0.2 "Minimum fan speed" annotation (Dialog(group="Setpoints"));
      parameter Modelica.SIunits.Temperature TSetCooUno = 273.15+30 "Unoccupied cooling setpoint" annotation (Dialog(group="Setpoints"));
      parameter Modelica.SIunits.Temperature TSetCooOcc = 273.15+24 "Occupied cooling setpoint" annotation (Dialog(group="Setpoints"));
      parameter Modelica.SIunits.Temperature TSetHeaUno = 273.15+15 "Unoccupied heating setpoint" annotation (Dialog(group="Setpoints"));
      parameter Modelica.SIunits.Temperature TSetHeaOcc = 273.15+21 "Occupied heating setpoint" annotation (Dialog(group="Setpoints"));
      parameter Modelica.SIunits.DimensionlessRatio kp = 0.1 "Controller P gain" annotation (Dialog(group="Gains"));
      parameter Modelica.SIunits.Time ki = 120 "Controller I gain" annotation (Dialog(group="Gains"));
      Modelica.Blocks.Interfaces.RealInput TZon "Measured zone air temperature"
        annotation (Placement(transformation(extent={{-140,-20},{-100,20}})));
      Modelica.Blocks.Interfaces.RealOutput yFan "Fan speed control signal"
        annotation (Placement(transformation(extent={{100,-10},{120,10}})));
      Modelica.Blocks.Interfaces.BooleanOutput yFanSta "Fan status control signal"
        annotation (Placement(transformation(extent={{100,-50},{120,-30}})));
      Buildings.Controls.Continuous.LimPID heaPI(
        controllerType=Modelica.Blocks.Types.SimpleController.PI,
        k=kp,
        Ti=ki) "Heating control signal"
        annotation (Placement(transformation(extent={{-10,30},{10,50}})));
      Buildings.Controls.Continuous.LimPID cooPI(
        controllerType=Modelica.Blocks.Types.SimpleController.PI,
        Ti=ki,
        reverseAction=true,
        k=kp,
        reset=Buildings.Types.Reset.Disabled) "Cooling control signal"
        annotation (Placement(transformation(extent={{-10,70},{10,90}})));
      Modelica.Blocks.Interfaces.RealOutput yCooVal
        "Control signal for cooling valve"
        annotation (Placement(transformation(extent={{100,70},{120,90}})));
      Buildings.Utilities.IO.SignalExchange.Read reaTSetCoo(y(unit="K"), description=
            "Zone air temperature setpoint for cooling")
        "Read zone cooling setpoint"
        annotation (Placement(transformation(extent={{-40,70},{-20,90}})));
      Buildings.Utilities.IO.SignalExchange.Read reaTSetHea(y(unit="K"), description=
            "Zone air temperature setpoint for heating")
                                                        "Read zone cooling heating"
        annotation (Placement(transformation(extent={{-40,30},{-20,50}})));
      Modelica.Blocks.Interfaces.RealOutput yHeaVal
        "Control signal for heating valve"
        annotation (Placement(transformation(extent={{100,30},{120,50}})));
      Modelica.Blocks.Math.Add add
        annotation (Placement(transformation(extent={{30,-10},{50,10}})));
      Modelica.Blocks.Logical.Hysteresis hys(uLow=1e-5, uHigh=minSpe)
        "Fan hysteresis"
        annotation (Placement(transformation(extent={{70,-50},{90,-30}})));

      Buildings.Utilities.IO.SignalExchange.Overwrite oveTSetCoo(u(
          unit="K",
          min=273.15 + 23,
          max=273.15 + 30), description="Zone temperature setpoint for cooling")
        "Overwrite for zone cooling setpoint"
        annotation (Placement(transformation(extent={{-70,70},{-50,90}})));
      Modelica.Blocks.Sources.CombiTimeTable TSetCoo(
        smoothness=Modelica.Blocks.Types.Smoothness.ConstantSegments,
        extrapolation=Modelica.Blocks.Types.Extrapolation.Periodic,
        table=[0,TSetCooUno; occSta,TSetCooOcc; occEnd,TSetCooUno; 24*3600,
            TSetCooUno]) "Cooling temperature setpoint for zone air"
        annotation (Placement(transformation(extent={{-100,70},{-80,90}})));
      Buildings.Utilities.IO.SignalExchange.Overwrite oveTSetHea(description="Zone temperature setpoint for heating",
          u(
          max=273.15 + 23,
          unit="K",
          min=273.15 + 15)) "Overwrite for zone heating setpoint"
        annotation (Placement(transformation(extent={{-70,30},{-50,50}})));
      Modelica.Blocks.Sources.CombiTimeTable TSetHea(
        smoothness=Modelica.Blocks.Types.Smoothness.ConstantSegments,
        extrapolation=Modelica.Blocks.Types.Extrapolation.Periodic,
        table=[0,TSetHeaUno; occSta,TSetHeaOcc; occEnd,TSetHeaUno; 24*3600,
            TSetHeaUno])
                     "Heating temperature setpoint for zone air"
        annotation (Placement(transformation(extent={{-100,30},{-80,50}})));
    equation
      connect(TZon, heaPI.u_m)
        annotation (Line(points={{-120,0},{0,0},{0,28}},     color={0,0,127}));
      connect(TZon, cooPI.u_m) annotation (Line(points={{-120,0},{-16,0},{-16,60},{0,
              60},{0,68}},   color={0,0,127}));
      connect(reaTSetCoo.y, cooPI.u_s)
        annotation (Line(points={{-19,80},{-12,80}}, color={0,0,127}));
      connect(reaTSetHea.y, heaPI.u_s)
        annotation (Line(points={{-19,40},{-12,40}}, color={0,0,127}));
      connect(cooPI.y, yCooVal) annotation (Line(points={{11,80},{110,80}},
                    color={0,0,127}));
      connect(heaPI.y, yHeaVal) annotation (Line(points={{11,40},{110,40}},
                    color={0,0,127}));
      connect(add.y, hys.u) annotation (Line(points={{51,0},{60,0},{60,-40},{68,-40}},
            color={0,0,127}));
      connect(cooPI.y, add.u1) annotation (Line(points={{11,80},{24,80},{24,6},{28,6}},
                   color={0,0,127}));
      connect(heaPI.y, add.u2) annotation (Line(points={{11,40},{20,40},{20,-6},{28,
              -6}},     color={0,0,127}));
      connect(hys.y, yFanSta)
        annotation (Line(points={{91,-40},{110,-40}}, color={255,0,255}));
      connect(add.y, yFan)
        annotation (Line(points={{51,0},{110,0}}, color={0,0,127}));
      connect(TSetHea.y[1], oveTSetHea.u)
        annotation (Line(points={{-79,40},{-72,40}}, color={0,0,127}));
      connect(TSetCoo.y[1], oveTSetCoo.u)
        annotation (Line(points={{-79,80},{-72,80}}, color={0,0,127}));
      connect(oveTSetCoo.y, reaTSetCoo.u)
        annotation (Line(points={{-49,80},{-42,80}}, color={0,0,127}));
      connect(oveTSetHea.y, reaTSetHea.u)
        annotation (Line(points={{-49,40},{-42,40}}, color={0,0,127}));
      annotation (Icon(coordinateSystem(preserveAspectRatio=false), graphics={
                                    Rectangle(
            extent={{-100,-100},{100,100}},
            lineColor={0,0,127},
            fillColor={255,255,255},
            fillPattern=FillPattern.Solid),
            Ellipse(
              extent={{-60,60},{62,-60}},
              lineColor={0,0,0},
              fillColor={0,140,72},
              fillPattern=FillPattern.Solid),
            Text(
              extent={{-24,24},{26,-16}},
              lineColor={255,255,255},
              fillColor={0,140,72},
              fillPattern=FillPattern.Solid,
              textStyle={TextStyle.Bold},
              textString="T"),              Text(
            extent={{-150,150},{150,110}},
            textString="%name",
            lineColor={0,0,255})}), Diagram(coordinateSystem(preserveAspectRatio=
                false)));
    end Thermostat;

    model InternalLoad "A model for internal loads"
      parameter Modelica.SIunits.HeatFlux senPower_nominal "Nominal sensible heat gain";
      parameter Modelica.SIunits.DimensionlessRatio radFraction "Fraction of sensible gain that is radiant";
      parameter Modelica.SIunits.HeatFlux latPower_nominal "Nominal latent heat gain";
      Modelica.Blocks.Sources.CombiTimeTable sch(
        extrapolation=Modelica.Blocks.Types.Extrapolation.Periodic,
        table=[0,0.1; 8*3600,0.1; 8*3600,1.0; 18*3600,1.0; 18*3600,0.1; 24*3600,0.1],
        columns={2})
        "Occupancy schedule"
        annotation (Placement(transformation(extent={{-100,-10},{-80,10}})));
      Modelica.Blocks.Interfaces.RealOutput rad "Radiant load in W/m^2"
        annotation (Placement(transformation(extent={{100,30},{120,50}})));
      Modelica.Blocks.Interfaces.RealOutput con "Convective load in W/m^2"
        annotation (Placement(transformation(extent={{100,-10},{120,10}})));
      Modelica.Blocks.Interfaces.RealOutput lat "Latent load in W/m^2"
        annotation (Placement(transformation(extent={{100,-50},{120,-30}})));
      Modelica.Blocks.Math.Gain gaiRad(k=senPower_nominal*radFraction)
        annotation (Placement(transformation(extent={{40,30},{60,50}})));
      Modelica.Blocks.Math.Gain gaiCon(k=senPower_nominal*(1 - radFraction))
        annotation (Placement(transformation(extent={{40,-10},{60,10}})));
      Modelica.Blocks.Math.Gain gaiLat(k=latPower_nominal)
        annotation (Placement(transformation(extent={{40,-50},{60,-30}})));
    equation
      connect(sch.y[1], gaiRad.u) annotation (Line(points={{-79,0},{-40,0},{-40,40},
              {38,40}}, color={0,0,127}));
      connect(sch.y[1], gaiCon.u)
        annotation (Line(points={{-79,0},{38,0}}, color={0,0,127}));
      connect(sch.y[1], gaiLat.u) annotation (Line(points={{-79,0},{-40,0},{-40,-40},
              {38,-40}}, color={0,0,127}));
      connect(gaiRad.y, rad)
        annotation (Line(points={{61,40},{110,40}}, color={0,0,127}));
      connect(gaiCon.y, con)
        annotation (Line(points={{61,0},{110,0}}, color={0,0,127}));
      connect(gaiLat.y, lat)
        annotation (Line(points={{61,-40},{110,-40}}, color={0,0,127}));
      annotation (Icon(coordinateSystem(preserveAspectRatio=false)), Diagram(
            coordinateSystem(preserveAspectRatio=false)));
    end InternalLoad;

    model OccupancyLoad
      "A model for occupancy and resulting internal loads"
      parameter Modelica.SIunits.Power senPower "Sensible heat gain per person";
      parameter Modelica.SIunits.DimensionlessRatio radFraction "Fraction of sensible gain that is radiant";
      parameter Modelica.SIunits.Power latPower "Latent heat gain per person";
      parameter Modelica.SIunits.MassFlowRate co2Gen "CO2 generation per person";
      parameter Modelica.SIunits.DimensionlessRatio occ_density "Number of occupants per m^2";
      Modelica.Blocks.Sources.CombiTimeTable sch(
        extrapolation=Modelica.Blocks.Types.Extrapolation.Periodic,
        table=[0,0; 8*3600,0; 8*3600,1.0; 18*3600,1.0; 18*3600,0; 24*3600,0],
        columns={2})
        "Occupancy schedule"
        annotation (Placement(transformation(extent={{-100,-10},{-80,10}})));
      Modelica.Blocks.Interfaces.RealOutput rad "Radiant load in W/m^2"
        annotation (Placement(transformation(extent={{100,30},{120,50}})));
      Modelica.Blocks.Interfaces.RealOutput con "Convective load in W/m^2"
        annotation (Placement(transformation(extent={{100,-10},{120,10}})));
      Modelica.Blocks.Interfaces.RealOutput lat "Latent load in W/m^2"
        annotation (Placement(transformation(extent={{100,-50},{120,-30}})));
      Modelica.Blocks.Math.Gain gaiRad(k=senPower*radFraction*occ_density)
        annotation (Placement(transformation(extent={{40,30},{60,50}})));
      Modelica.Blocks.Math.Gain gaiCon(k=senPower*(1 - radFraction)*occ_density)
        annotation (Placement(transformation(extent={{40,-10},{60,10}})));
      Modelica.Blocks.Math.Gain gaiLat(k=latPower*occ_density)
        annotation (Placement(transformation(extent={{40,-50},{60,-30}})));
      Modelica.Blocks.Math.Gain gaiCO2(k=co2Gen*occ_density)
        annotation (Placement(transformation(extent={{40,-90},{60,-70}})));
      Modelica.Blocks.Interfaces.RealOutput co2 "CO2 generation in kg/s/m^2"
        annotation (Placement(transformation(extent={{100,-90},{120,-70}})));
    equation
      connect(sch.y[1], gaiRad.u) annotation (Line(points={{-79,0},{-40,0},{-40,40},
              {38,40}}, color={0,0,127}));
      connect(sch.y[1], gaiCon.u)
        annotation (Line(points={{-79,0},{38,0}}, color={0,0,127}));
      connect(sch.y[1], gaiLat.u) annotation (Line(points={{-79,0},{-40,0},{-40,-40},
              {38,-40}}, color={0,0,127}));
      connect(gaiRad.y, rad)
        annotation (Line(points={{61,40},{110,40}}, color={0,0,127}));
      connect(gaiCon.y, con)
        annotation (Line(points={{61,0},{110,0}}, color={0,0,127}));
      connect(gaiLat.y, lat)
        annotation (Line(points={{61,-40},{110,-40}}, color={0,0,127}));
      connect(gaiCO2.y, co2)
        annotation (Line(points={{61,-80},{110,-80}}, color={0,0,127}));
      connect(sch.y[1], gaiCO2.u) annotation (Line(points={{-79,0},{-40,0},{-40,-80},
              {38,-80}}, color={0,0,127}));
      annotation (Icon(coordinateSystem(preserveAspectRatio=false)), Diagram(
            coordinateSystem(preserveAspectRatio=false)));
    end OccupancyLoad;

    model FanControl "Internal fan controller to limit minimum speed"
        parameter Modelica.SIunits.DimensionlessRatio minSpe = 0.2 "Minimum fan speed";
      Modelica.Blocks.Nonlinear.Limiter lim(uMax=1, uMin=minSpe)
        "Fan speed limiter"
        annotation (Placement(transformation(extent={{-20,20},{0,40}})));
      Modelica.Blocks.Logical.Switch swiFan "Fan enable switch"
        annotation (Placement(transformation(extent={{20,-10},{40,10}})));
      Modelica.Blocks.Sources.Constant off(k=0) "Off signal"
        annotation (Placement(transformation(extent={{-20,-40},{0,-20}})));
      Modelica.Blocks.Interfaces.RealOutput yFan "Fan speed control signal"
        annotation (Placement(transformation(extent={{100,-10},{120,10}})));
      Modelica.Blocks.Interfaces.RealInput uFan "Fan speed control signal"
        annotation (Placement(transformation(extent={{-140,20},{-100,60}})));
      Modelica.Blocks.Interfaces.BooleanInput uFanSta "Fan status control signal"
        annotation (Placement(transformation(extent={{-140,-40},{-100,0}})));
    equation
      connect(lim.y, swiFan.u1)
        annotation (Line(points={{1,30},{10,30},{10,8},{18,8}}, color={0,0,127}));
      connect(swiFan.y, yFan)
        annotation (Line(points={{41,0},{110,0}}, color={0,0,127}));
      connect(off.y, swiFan.u3) annotation (Line(points={{1,-30},{10,-30},{10,-8},{18,
              -8}}, color={0,0,127}));
      connect(lim.u, uFan) annotation (Line(points={{-22,30},{-60,30},{-60,40},{-120,
              40}}, color={0,0,127}));
      connect(swiFan.u2, uFanSta) annotation (Line(points={{18,0},{-60,0},{-60,-20},
              {-120,-20}}, color={255,0,255}));
      annotation (Icon(coordinateSystem(preserveAspectRatio=false)), Diagram(
            coordinateSystem(preserveAspectRatio=false)));
    end FanControl;

    model FanCoilUnit_T
      "Four-pipe fan coil unit model with direct temperature input"
      replaceable package Medium1 = Buildings.Media.Air(extraPropertiesNames={"CO2"});
      parameter Modelica.SIunits.MassFlowRate mAir_flow_nominal=0.55 "Nominal air flowrate" annotation (Dialog(group="Air"));
      parameter Modelica.SIunits.DimensionlessRatio COP = 3 "Assumed COP of chiller supplying chilled water to FCU in [W_thermal/W_electric]" annotation (Dialog(group="Plant"));
      parameter Modelica.SIunits.DimensionlessRatio eff = 0.9 "Assumed efficiency of gas boiler supplying hot water to FCU in [W_gas/W_thermal]" annotation (Dialog(group="Plant"));
      final parameter Modelica.SIunits.Pressure dpAir_nominal=185 "Nominal supply air pressure";
      Modelica.Fluid.Interfaces.FluidPort_a returnAir(redeclare final package
          Medium = Medium1) "Return air" annotation (Placement(transformation(
              extent={{130,-170},{150,-150}}),
                                            iconTransformation(extent={{130,-170},{
                150,-150}})));
      Modelica.Fluid.Interfaces.FluidPort_b supplyAir(redeclare final package
          Medium = Medium1) "Supply air"
        annotation (Placement(transformation(extent={{130,90},{150,110}}),
            iconTransformation(extent={{130,90},{150,110}})));

      Modelica.Blocks.Interfaces.RealInput uFan "Fan speed signal"
        annotation (Placement(transformation(extent={{-180,-60},{-140,-20}})));
      Buildings.Fluid.Sensors.MassFlowRate senSupFlo(redeclare package Medium
          = Medium1) "Supply flow meter"
        annotation (Placement(transformation(extent={{20,90},{40,110}})));
      Modelica.Blocks.Interfaces.RealOutput PFan "Fan electrical power consumption"
        annotation (Placement(transformation(extent={{140,130},{160,150}})));
      Modelica.Blocks.Interfaces.RealOutput PHea "Heating power"
        annotation (Placement(transformation(extent={{140,150},{160,170}})));
      Modelica.Blocks.Interfaces.RealOutput PCoo "Cooling power"
        annotation (Placement(transformation(extent={{140,170},{160,190}})));
      Modelica.Blocks.Math.Gain powHea(k=1/eff)
        annotation (Placement(transformation(extent={{-8,150},{12,170}})));
      Modelica.Blocks.Math.Gain powCoo(k=1/COP)
        annotation (Placement(transformation(extent={{-8,170},{12,190}})));
      Buildings.Utilities.IO.SignalExchange.Read reaFloSup(y(unit="kg/s"), description=
            "Supply air mass flow rate") "Read supply air flowrate"
        annotation (Placement(transformation(extent={{40,110},{60,130}})));
      Modelica.Blocks.Interfaces.RealInput TSup "Temperature of supply air"
        annotation (Placement(transformation(extent={{-180,20},{-140,60}})));
      Buildings.Utilities.IO.SignalExchange.Read reaPCoo(
        y(unit="W"),
        KPIs=Buildings.Utilities.IO.SignalExchange.SignalTypes.SignalsForKPIs.ElectricPower,
        description="Cooling electrical power consumption")
        "Read power for cooling"
        annotation (Placement(transformation(extent={{70,170},{90,190}})));

      Buildings.Utilities.IO.SignalExchange.Read reaPHea(
        y(unit="W"),
        KPIs=Buildings.Utilities.IO.SignalExchange.SignalTypes.SignalsForKPIs.GasPower,
        description="Heating thermal power consumption") "Read power for heating"
        annotation (Placement(transformation(extent={{70,150},{90,170}})));

      Buildings.Utilities.IO.SignalExchange.Read reaPFan(
        y(unit="W"),
        description="Supply fan electrical power consumption",
        KPIs=Buildings.Utilities.IO.SignalExchange.SignalTypes.SignalsForKPIs.ElectricPower)
        "Read power for supply fan"
        annotation (Placement(transformation(extent={{70,130},{90,150}})));
      Modelica.Blocks.Math.Gain fanGai(k=mAir_flow_nominal) "Fan gain"
        annotation (Placement(transformation(extent={{-40,-50},{-20,-30}})));
      Buildings.Fluid.Movers.FlowControlled_m_flow fan(
        redeclare package Medium = Medium1,
        m_flow_nominal=mAir_flow_nominal,
        addPowerToMedium=false,
        nominalValuesDefineDefaultPressureCurve=true,
        dp_nominal=dpAir_nominal) "Supply fan"
        annotation (Placement(transformation(extent={{-10,-10},{10,10}},
            rotation=90,
            origin={12,-40})));
      Buildings.Fluid.FixedResistances.PressureDrop res(
        redeclare package Medium = Medium1,
        m_flow_nominal=mAir_flow_nominal,
        dp_nominal=dpAir_nominal) "Air system resistance"
        annotation (Placement(transformation(extent={{-10,-10},{10,10}},
            rotation=90,
            origin={10,50})));
      Modelica.Blocks.Sources.RealExpression powCooThe(y=max(0, -senSupFlo.m_flow*(
            supplyAir.h_outflow - inStream(returnAir.h_outflow))))
                                                         "Cooling thermal power"
        annotation (Placement(transformation(extent={{-100,170},{-80,190}})));
      Modelica.Blocks.Sources.RealExpression powHeaThe(y=max(0, senSupFlo.m_flow*(
            supplyAir.h_outflow - inStream(returnAir.h_outflow))))
                                                         "Heating thermal power"
        annotation (Placement(transformation(extent={{-100,150},{-80,170}})));
      Buildings.Fluid.Sources.PropertySource_T coi(use_T_in=true, redeclare
          package Medium =
                   Medium1) "Cooling and heating coil" annotation (Placement(
            transformation(
            extent={{-10,-10},{10,10}},
            rotation=90,
            origin={10,0})));
      Buildings.Utilities.IO.SignalExchange.Overwrite oveTSup(description=
            "Supply air temperature setpoint", u(
          min=285.15,
          max=313.15,
          unit="K")) "Overwrite for supply air temperature signal"
        annotation (Placement(transformation(extent={{-120,30},{-100,50}})));
      Buildings.Utilities.IO.SignalExchange.Overwrite oveFan(description=
            "Fan control signal as air mass flow rate normalized to the design air mass flow rate",
                                        u(
          min=0,
          max=1,
          unit="1")) "Overwrite for fan control signal"
        annotation (Placement(transformation(extent={{-120,-50},{-100,-30}})));
    equation
      connect(senSupFlo.m_flow, reaFloSup.u)
        annotation (Line(points={{30,111},{30,120},{38,120}},
                                                           color={0,0,127}));
      connect(powCoo.y, reaPCoo.u)
        annotation (Line(points={{13,180},{68,180}}, color={0,0,127}));
      connect(reaPCoo.y, PCoo)
        annotation (Line(points={{91,180},{150,180}}, color={0,0,127}));
      connect(powHea.y, reaPHea.u)
        annotation (Line(points={{13,160},{68,160}}, color={0,0,127}));
      connect(reaPHea.y, PHea)
        annotation (Line(points={{91,160},{150,160}}, color={0,0,127}));
      connect(reaPFan.y, PFan)
        annotation (Line(points={{91,140},{150,140}}, color={0,0,127}));
      connect(fanGai.y, fan.m_flow_in)
        annotation (Line(points={{-19,-40},{0,-40}},          color={0,0,127}));
      connect(reaPFan.u, fan.P)
        annotation (Line(points={{68,140},{4,140},{4,-29},{3,-29}},
                                                                color={0,0,127}));
      connect(res.port_b, senSupFlo.port_a)
        annotation (Line(points={{10,60},{10,100},{20,100}}, color={0,127,255}));
      connect(powCooThe.y, powCoo.u)
        annotation (Line(points={{-79,180},{-10,180}}, color={0,0,127}));
      connect(powHeaThe.y, powHea.u)
        annotation (Line(points={{-79,160},{-10,160}}, color={0,0,127}));
      connect(senSupFlo.port_b, supplyAir)
        annotation (Line(points={{40,100},{140,100}}, color={0,127,255}));
      connect(fan.port_b, coi.port_a)
        annotation (Line(points={{12,-30},{10,-30},{10,-10}}, color={0,127,255}));
      connect(coi.port_b, res.port_a)
        annotation (Line(points={{10,10},{10,40}}, color={0,127,255}));
      connect(fan.port_a, returnAir) annotation (Line(points={{12,-50},{12,-160},{
              140,-160}}, color={0,127,255}));
      connect(TSup, oveTSup.u)
        annotation (Line(points={{-160,40},{-122,40}}, color={0,0,127}));
      connect(uFan, oveFan.u)
        annotation (Line(points={{-160,-40},{-122,-40}}, color={0,0,127}));
      connect(oveTSup.y, coi.T_in) annotation (Line(points={{-99,40},{-60,40},{-60,
              -4},{-2,-4}}, color={0,0,127}));
      connect(oveFan.y, fanGai.u)
        annotation (Line(points={{-99,-40},{-42,-40}}, color={0,0,127}));
      annotation (Icon(coordinateSystem(preserveAspectRatio=false, extent={{-140,
                -180},{140,180}}),                                  graphics={
                                            Text(
            extent={{-150,184},{150,144}},
            textString="%name",
            lineColor={0,0,255}), Rectangle(
              extent={{-140,180},{140,-180}},
              lineColor={0,0,0},
              fillColor={215,215,215},
              fillPattern=FillPattern.Solid),
                                            Text(
            extent={{-150,238},{150,198}},
            textString="%name",
            lineColor={0,0,255})}),                                  Diagram(
            coordinateSystem(preserveAspectRatio=false, extent={{-140,-180},{140,
                180}})),
        experiment(
          StartTime=20736000,
          StopTime=21600000,
          Interval=599.999616,
          Tolerance=1e-06,
          __Dymola_Algorithm="Cvode"));
    end FanCoilUnit_T;

    model Thermostat_T
      "Implements basic control of FCU to maintain zone air temperature with temperature as output"
      parameter Modelica.SIunits.Time occSta = 8*3600 "Occupancy start time" annotation (Dialog(group="Schedule"));
      parameter Modelica.SIunits.Time occEnd = 18*3600 "Occupancy end time" annotation (Dialog(group="Schedule"));
      parameter Modelica.SIunits.DimensionlessRatio minSpe = 0.2 "Minimum fan speed" annotation (Dialog(group="Setpoints"));
      parameter Modelica.SIunits.Temperature TSetCooUno = 273.15+30 "Unoccupied cooling setpoint" annotation (Dialog(group="Setpoints"));
      parameter Modelica.SIunits.Temperature TSetCooOcc = 273.15+24 "Occupied cooling setpoint" annotation (Dialog(group="Setpoints"));
      parameter Modelica.SIunits.Temperature TSetHeaUno = 273.15+15 "Unoccupied heating setpoint" annotation (Dialog(group="Setpoints"));
      parameter Modelica.SIunits.Temperature TSetHeaOcc = 273.15+21 "Occupied heating setpoint" annotation (Dialog(group="Setpoints"));
      parameter Modelica.SIunits.DimensionlessRatio kp = 0.1 "Controller P gain" annotation (Dialog(group="Gains"));
      parameter Modelica.SIunits.Time ki = 120 "Controller I gain" annotation (Dialog(group="Gains"));
      Modelica.Blocks.Interfaces.RealInput TZon "Measured zone air temperature"
        annotation (Placement(transformation(extent={{-140,-20},{-100,20}})));
      Modelica.Blocks.Interfaces.RealOutput yFan "Fan speed control signal"
        annotation (Placement(transformation(extent={{100,-10},{120,10}})));
      Buildings.Controls.Continuous.LimPID heaPI(
        controllerType=Modelica.Blocks.Types.SimpleController.PI,
        k=kp,
        Ti=ki) "Heating control signal"
        annotation (Placement(transformation(extent={{-10,30},{10,50}})));
      Buildings.Controls.Continuous.LimPID cooPI(
        controllerType=Modelica.Blocks.Types.SimpleController.PI,
        Ti=ki,
        k=kp,
        reverseActing=false,
        reset=Buildings.Types.Reset.Disabled) "Cooling control signal"
        annotation (Placement(transformation(extent={{-10,70},{10,90}})));
      Modelica.Blocks.Math.Add add
        annotation (Placement(transformation(extent={{40,-10},{60,10}})));

      Buildings.Utilities.IO.SignalExchange.Overwrite oveTSetCoo(u(
          unit="K",
          min=273.15 + 23,
          max=273.15 + 30), description="Zone temperature setpoint for cooling")
        "Overwrite for zone cooling setpoint"
        annotation (Placement(transformation(extent={{-70,70},{-50,90}})));
      Modelica.Blocks.Sources.CombiTimeTable TSetCoo(
        smoothness=Modelica.Blocks.Types.Smoothness.ConstantSegments,
        extrapolation=Modelica.Blocks.Types.Extrapolation.Periodic,
        table=[0,TSetCooUno; occSta,TSetCooOcc; occEnd,TSetCooUno; 24*3600,
            TSetCooUno]) "Cooling temperature setpoint for zone air"
        annotation (Placement(transformation(extent={{-100,70},{-80,90}})));
      Buildings.Utilities.IO.SignalExchange.Overwrite oveTSetHea(description="Zone temperature setpoint for heating",
          u(
          max=273.15 + 23,
          unit="K",
          min=273.15 + 15)) "Overwrite for zone heating setpoint"
        annotation (Placement(transformation(extent={{-70,30},{-50,50}})));
      Modelica.Blocks.Sources.CombiTimeTable TSetHea(
        smoothness=Modelica.Blocks.Types.Smoothness.ConstantSegments,
        extrapolation=Modelica.Blocks.Types.Extrapolation.Periodic,
        table=[0,TSetHeaUno; occSta,TSetHeaOcc; occEnd,TSetHeaUno; 24*3600,
            TSetHeaUno])
                     "Heating temperature setpoint for zone air"
        annotation (Placement(transformation(extent={{-100,30},{-80,50}})));
      ControlToTemperature cooTem(
        THig=TSetCooOcc,
        TLow=285.15,
        reverseAction=true) "Convert to cooling temperature signal"
        annotation (Placement(transformation(extent={{40,80},{60,100}})));
      ControlToTemperature heaTem(
        THig=313.15,
        TLow=TSetHeaOcc,
        reverseAction=false) "Convert to heating temperature signal"
        annotation (Placement(transformation(extent={{40,20},{60,40}})));
      Modelica.Blocks.Interfaces.RealOutput TSup "Supply temperature setpoint"
        annotation (Placement(transformation(extent={{100,50},{120,70}})));
      Modelica.Blocks.Logical.Switch swi "Cooling and heating switch"
        annotation (Placement(transformation(extent={{70,50},{90,70}})));
      Modelica.Blocks.Math.Feedback errCoo
        "Control error on room temperature for cooling"
        annotation (Placement(transformation(extent={{-58,-30},{-38,-10}})));
      Modelica.Blocks.Logical.Hysteresis hysCoo(uLow=-1, uHigh=0)
        "Hysteresis for cooling signal"
        annotation (Placement(transformation(extent={{-20,-30},{0,-10}})));
      Modelica.Blocks.Nonlinear.Limiter fanLim(uMax=1, uMin=0)
        "Limit fan signal to between 0 and 1"
        annotation (Placement(transformation(extent={{70,-10},{90,10}})));
    equation
      connect(TZon, heaPI.u_m)
        annotation (Line(points={{-120,0},{0,0},{0,28}},     color={0,0,127}));
      connect(TZon, cooPI.u_m) annotation (Line(points={{-120,0},{-16,0},{-16,60},{0,
              60},{0,68}},   color={0,0,127}));
      connect(cooPI.y, add.u1) annotation (Line(points={{11,80},{30,80},{30,6},{38,
              6}}, color={0,0,127}));
      connect(heaPI.y, add.u2) annotation (Line(points={{11,40},{20,40},{20,-6},{38,
              -6}},     color={0,0,127}));
      connect(TSetHea.y[1], oveTSetHea.u)
        annotation (Line(points={{-79,40},{-72,40}}, color={0,0,127}));
      connect(TSetCoo.y[1], oveTSetCoo.u)
        annotation (Line(points={{-79,80},{-72,80}}, color={0,0,127}));
      connect(cooPI.y, cooTem.u) annotation (Line(points={{11,80},{20,80},{20,90},{
              38,90}}, color={0,0,127}));
      connect(heaPI.y, heaTem.u) annotation (Line(points={{11,40},{20,40},{20,30},{
              38,30}}, color={0,0,127}));
      connect(swi.y, TSup)
        annotation (Line(points={{91,60},{110,60}}, color={0,0,127}));
      connect(cooTem.T, swi.u1)
        annotation (Line(points={{61,90},{68,90},{68,68}}, color={0,0,127}));
      connect(heaTem.T, swi.u3)
        annotation (Line(points={{61,30},{68,30},{68,52}}, color={0,0,127}));
      connect(errCoo.y, hysCoo.u)
        annotation (Line(points={{-39,-20},{-22,-20}}, color={0,0,127}));
      connect(hysCoo.y, swi.u2) annotation (Line(points={{1,-20},{64,-20},{64,60},{
              68,60}}, color={255,0,255}));
      connect(TZon, errCoo.u1) annotation (Line(points={{-120,0},{-80,0},{-80,-20},
              {-56,-20}}, color={0,0,127}));
      connect(TSetCoo.y[1], errCoo.u2) annotation (Line(points={{-79,80},{-76,80},{
              -76,-40},{-48,-40},{-48,-28}}, color={0,0,127}));
      connect(add.y, fanLim.u)
        annotation (Line(points={{61,0},{68,0}}, color={0,0,127}));
      connect(fanLim.y, yFan)
        annotation (Line(points={{91,0},{110,0}}, color={0,0,127}));
      connect(oveTSetHea.y, heaPI.u_s)
        annotation (Line(points={{-49,40},{-12,40}}, color={0,0,127}));
      connect(oveTSetCoo.y, cooPI.u_s)
        annotation (Line(points={{-49,80},{-12,80}}, color={0,0,127}));
      annotation (Icon(coordinateSystem(preserveAspectRatio=false), graphics={
                                    Rectangle(
            extent={{-100,-100},{100,100}},
            lineColor={0,0,127},
            fillColor={255,255,255},
            fillPattern=FillPattern.Solid),
            Ellipse(
              extent={{-60,60},{62,-60}},
              lineColor={0,0,0},
              fillColor={0,140,72},
              fillPattern=FillPattern.Solid),
            Text(
              extent={{-24,24},{26,-16}},
              lineColor={255,255,255},
              fillColor={0,140,72},
              fillPattern=FillPattern.Solid,
              textStyle={TextStyle.Bold},
              textString="T"),              Text(
            extent={{-150,150},{150,110}},
            textString="%name",
            lineColor={0,0,255})}), Diagram(coordinateSystem(preserveAspectRatio=
                false)));
    end Thermostat_T;

    model ControlToTemperature
      "Convert control signals to temperature setpoints"
      parameter Modelica.SIunits.Temperature THig "High temperature";
      parameter Modelica.SIunits.Temperature TLow "Low temperature";
      parameter Boolean reverseAction = false "True in the case of cooling";
      Modelica.Blocks.Interfaces.RealInput u "Control signal"
        annotation (Placement(transformation(extent={{-140,-20},{-100,20}})));
      Modelica.Blocks.Interfaces.RealOutput T "Supply air temperature"
        annotation (Placement(transformation(extent={{100,-10},{120,10}})));
      Modelica.Blocks.Math.Gain gain(k=THig - TLow)
        annotation (Placement(transformation(extent={{-20,10},{0,30}})));
      Modelica.Blocks.Math.Add add(k1=if reverseAction then -1 else 1)
        annotation (Placement(transformation(extent={{20,-10},{40,10}})));
      Modelica.Blocks.Sources.Constant const(k=if reverseAction then THig else TLow)
        annotation (Placement(transformation(extent={{-20,-30},{0,-10}})));
    equation
      connect(gain.y, add.u1)
        annotation (Line(points={{1,20},{10,20},{10,6},{18,6}}, color={0,0,127}));
      connect(add.y, T) annotation (Line(points={{41,0},{110,0}}, color={0,0,127}));
      connect(const.y, add.u2) annotation (Line(points={{1,-20},{10,-20},{10,-6},{18,
              -6}}, color={0,0,127}));
      connect(u, gain.u) annotation (Line(points={{-120,0},{-40,0},{-40,20},{-22,20}},
            color={0,0,127}));
      annotation (Icon(coordinateSystem(preserveAspectRatio=false)), Diagram(
            coordinateSystem(preserveAspectRatio=false)));
    end ControlToTemperature;
  end BaseClasses;
annotation (uses(Modelica(version="3.2.3"),
      Buildings(version="8.0.1")),
  version="1",
  conversion(noneFromVersion=""));
end SingleZoneFanCoilUnit;
