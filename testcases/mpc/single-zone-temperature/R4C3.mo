within ;
model R4C3
  Modelica.Thermal.HeatTransfer.Components.HeatCapacitor CWalExt(C=Cwe)
    annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=180,
        origin={-20,-32})));
  Modelica.Thermal.HeatTransfer.Components.ThermalResistor RExt(R=Re)
    annotation (Placement(transformation(extent={{-50,-10},{-30,10}})));
  Modelica.Thermal.HeatTransfer.Sources.PrescribedTemperature preTem
    annotation (Placement(transformation(extent={{-88,-10},{-68,10}})));
  Modelica.Blocks.Interfaces.RealInput To
    annotation (Placement(transformation(extent={{-140,-20},{-100,20}})));
  Modelica.Thermal.HeatTransfer.Components.ThermalResistor RWal(R=Rw)
    annotation (Placement(transformation(extent={{-8,-10},{12,10}})));
  Modelica.Thermal.HeatTransfer.Components.HeatCapacitor CWalInt(C=Cwi)
    annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=180,
        origin={20,-32})));
  Modelica.Thermal.HeatTransfer.Components.ThermalResistor RInt(R=Ri)
    annotation (Placement(transformation(extent={{30,-10},{50,10}})));
  Modelica.Thermal.HeatTransfer.Components.ThermalResistor RGla(R=Rg)
    annotation (Placement(transformation(extent={{-10,30},{10,50}})));
  Modelica.Thermal.HeatTransfer.Components.HeatCapacitor CAir(C=Cai)
    annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=180,
        origin={64,-10})));
  Modelica.Thermal.HeatTransfer.Sources.PrescribedHeatFlow preQSolExt
    annotation (Placement(transformation(extent={{-70,50},{-50,70}})));
  Modelica.Blocks.Interfaces.RealInput qrad_e
    annotation (Placement(transformation(extent={{-140,40},{-100,80}})));
  Modelica.Blocks.Interfaces.RealInput qrad_i
    annotation (Placement(transformation(extent={{-140,70},{-100,110}})));
  Modelica.Thermal.HeatTransfer.Sources.PrescribedHeatFlow preQSolInt
    annotation (Placement(transformation(extent={{-70,80},{-50,100}})));
  Modelica.Blocks.Interfaces.RealInput qhvac "negative for cooling"
    annotation (Placement(transformation(extent={{-140,-110},{-100,-70}})));
  Modelica.Thermal.HeatTransfer.Sources.PrescribedHeatFlow preQHVAC
    annotation (Placement(transformation(extent={{-72,-100},{-52,-80}})));
  Modelica.Thermal.HeatTransfer.Sources.PrescribedHeatFlow preQConInt
    annotation (Placement(transformation(extent={{-72,-70},{-52,-50}})));
  Modelica.Blocks.Interfaces.RealInput qcon_i "convective heat gain internal"
    annotation (Placement(transformation(extent={{-140,-80},{-100,-40}})));
  Modelica.Thermal.HeatTransfer.Sensors.TemperatureSensor temSen
    annotation (Placement(transformation(extent={{76,30},{96,50}})));
  Modelica.Blocks.Interfaces.RealOutput Ti
    "Absolute temperature as output signal"
    annotation (Placement(transformation(extent={{100,30},{120,50}})));
  parameter Modelica.SIunits.ThermalResistance Rg
    "Constant thermal resistance of material";
  parameter Modelica.SIunits.ThermalResistance Re
    "Constant thermal resistance of material";
  parameter Modelica.SIunits.ThermalResistance Rw
    "Constant thermal resistance of material";
  parameter Modelica.SIunits.ThermalResistance Ri
    "Constant thermal resistance of material";
  parameter Modelica.SIunits.HeatCapacity Cwe
    "Heat capacity of element (= cp*m)";
  parameter Modelica.SIunits.HeatCapacity Cwi
    "Heat capacity of element (= cp*m)";
  parameter Modelica.SIunits.HeatCapacity Cai
    "Heat capacity of element (= cp*m)";
equation
  connect(preTem.T, To)
    annotation (Line(points={{-90,0},{-120,0}}, color={0,0,127}));
  connect(preTem.port, RExt.port_a)
    annotation (Line(points={{-68,0},{-50,0}}, color={191,0,0}));
  connect(RExt.port_b, CWalExt.port)
    annotation (Line(points={{-30,0},{-20,0},{-20,-22}}, color={191,0,0}));
  connect(RWal.port_a, CWalExt.port)
    annotation (Line(points={{-8,0},{-20,0},{-20,-22}}, color={191,0,0}));
  connect(RWal.port_b, CWalInt.port)
    annotation (Line(points={{12,0},{20,0},{20,-22}}, color={191,0,0}));
  connect(RInt.port_a, CWalInt.port)
    annotation (Line(points={{30,0},{20,0},{20,-22}}, color={191,0,0}));
  connect(preTem.port, RGla.port_a) annotation (Line(points={{-68,0},{-60,0},{
          -60,40},{-10,40}}, color={191,0,0}));
  connect(RGla.port_b, CAir.port)
    annotation (Line(points={{10,40},{64,40},{64,0}}, color={191,0,0}));
  connect(RInt.port_b, CAir.port)
    annotation (Line(points={{50,0},{64,0}}, color={191,0,0}));
  connect(preQSolExt.Q_flow, qrad_e)
    annotation (Line(points={{-70,60},{-120,60}}, color={0,0,127}));
  connect(preQSolExt.port, CWalExt.port)
    annotation (Line(points={{-50,60},{-20,60},{-20,-22}}, color={191,0,0}));
  connect(qrad_i, preQSolInt.Q_flow)
    annotation (Line(points={{-120,90},{-70,90}}, color={0,0,127}));
  connect(preQSolInt.port, CWalInt.port)
    annotation (Line(points={{-50,90},{20,90},{20,-22}}, color={191,0,0}));
  connect(qhvac, preQHVAC.Q_flow)
    annotation (Line(points={{-120,-90},{-72,-90}}, color={0,0,127}));
  connect(qcon_i, preQConInt.Q_flow)
    annotation (Line(points={{-120,-60},{-72,-60}}, color={0,0,127}));
  connect(preQConInt.port, CAir.port) annotation (Line(points={{-52,-60},{80,
          -60},{80,0},{64,0}}, color={191,0,0}));
  connect(preQHVAC.port, CAir.port) annotation (Line(points={{-52,-90},{80,-90},
          {80,0},{64,0}}, color={191,0,0}));
  connect(temSen.port, CAir.port)
    annotation (Line(points={{76,40},{66,40},{66,0},{64,0}}, color={191,0,0}));
  connect(temSen.T, Ti)
    annotation (Line(points={{96,40},{110,40}}, color={0,0,127}));
  annotation (
    Icon(coordinateSystem(preserveAspectRatio=false)),
    Diagram(coordinateSystem(preserveAspectRatio=false)),
    uses(Modelica(version="3.2.3")));
end R4C3;
