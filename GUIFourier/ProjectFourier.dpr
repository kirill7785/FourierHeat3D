program ProjectFourier;

uses
  Vcl.Forms,
  Main1 in 'Main1.pas' {FormTopology},
  UnitLayers2 in 'UnitLayers2.pas' {FormLayers},
  UnitConditions in 'UnitConditions.pas' {FormConditions},
  UnitExport in 'UnitExport.pas' {FormExport},
  UnitMatherials in 'UnitMatherials.pas' {FormMatherials},
  UnitLauncher in 'UnitLauncher.pas' {FormLauncher};

{$R *.res}

begin
  Application.Initialize;
  Application.MainFormOnTaskbar := True;
  Application.CreateForm(TFormTopology, FormTopology);
  Application.CreateForm(TFormLayers, FormLayers);
  Application.CreateForm(TFormConditions, FormConditions);
  Application.CreateForm(TFormExport, FormExport);
  Application.CreateForm(TFormMatherials, FormMatherials);
  Application.CreateForm(TFormLauncher, FormLauncher);
  Application.Run;
end.
