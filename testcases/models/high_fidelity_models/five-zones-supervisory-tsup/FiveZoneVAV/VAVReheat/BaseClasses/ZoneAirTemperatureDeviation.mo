within FiveZoneVAV.VAVReheat.BaseClasses;
partial model ZoneAirTemperatureDeviation
  "Calculate the zone air temperature deviation outside the boundary"

   FiveZoneVAV.VAVReheat.BaseClasses.BandDeviationSum banDevSum[5](each
      uppThreshold=25 + 273.15, each lowThreshold=23 + 273.15)
    annotation (Placement(transformation(extent={{1240,480},{1260,500}})));
  Modelica.Blocks.Math.MultiSum TAirDev(nu=5) "Zone air temperature deviation in total"
    annotation (Placement(transformation(extent={{1282,484},{1294,496}})));
  Modelica.Blocks.Continuous.Integrator TAirTotDev
    annotation (Placement(transformation(extent={{1318,480},{1338,500}})));
equation
  connect(TAirDev.y,TAirTotDev. u)
    annotation (Line(points={{1295.02,490},{1316,490}}, color={0,0,127}));
  connect(banDevSum.TDev,TAirDev. u[1:5]) annotation (Line(points={{1261,490},
          {1272,490},{1272,486.64},{1282,486.64}}, color={0,0,127}));
  annotation (Diagram(coordinateSystem(extent={{-100,-100},{1580,700}})), Icon(
        coordinateSystem(extent={{-100,-100},{1580,700}})));
end ZoneAirTemperatureDeviation;
