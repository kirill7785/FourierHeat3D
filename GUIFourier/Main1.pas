unit Main1;

interface

uses
  Winapi.Windows, Winapi.Messages, System.SysUtils, System.Variants, System.Classes, Vcl.Graphics,
  Vcl.Controls, Vcl.Forms, Vcl.Dialogs, Vcl.StdCtrls, Vcl.Menus, ShellAPI;

  type
  TMatherial = record
    density : Real; // плотность
    heatCapasity : Real; // Теплоёмкость при постоянном давлении
    thermalConductivity : Real; // теплопроводность
    multiplyerConductivityPlane : Real;
    multiplyerConductivityNormal : Real;
    alphaForTemperatureDepend : Real;
    name : string;
  end;

type
  TFormTopology = class(TForm)
    Label1: TLabel;
    LabelThermalresistance: TLabel;
    Label3: TLabel;
    Button1: TButton;
    Labelwait: TLabel;
    Edit1: TEdit;
    Label2: TLabel;
    Label4: TLabel;
    Label5: TLabel;
    Edit2: TEdit;
    Label6: TLabel;
    Label7: TLabel;
    ComboBox1: TComboBox;
    Label8: TLabel;
    Edit3: TEdit;
    Label9: TLabel;
    Label10: TLabel;
    Edit4: TEdit;
    Label11: TLabel;
    Label12: TLabel;
    ComboBox2: TComboBox;
    Button2: TButton;
    Label13: TLabel;
    Label14: TLabel;
    ComboBox3: TComboBox;
    Edit5: TEdit;
    Label15: TLabel;
    MainMenu1: TMainMenu;
    File1: TMenuItem;
    File2: TMenuItem;
    Define1: TMenuItem;
    Define2: TMenuItem;
    Export1: TMenuItem;
    Matherials1: TMenuItem;
    Close1: TMenuItem;
    Launcher1: TMenuItem;
    procedure Button1Click(Sender: TObject);
    procedure Button2Click(Sender: TObject);
    procedure Define2Click(Sender: TObject);
    procedure Export1Click(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure Matherials1Click(Sender: TObject);
    procedure Close1Click(Sender: TObject);
    procedure Launcher1Click(Sender: TObject);
  private
    { Private declarations }
    bFirst_Matherial : Boolean;
  public
    { Public declarations }
     matherial : array of TMatherial;
  end;

var
  FormTopology: TFormTopology;

implementation

uses UnitLayers2, UnitConditions, UnitExport, UnitMatherials, UnitLauncher;
     procedure fourier_solve(var thermal_resistance : Real; size_x : Real; size_y : Real; distance_x : Real; distance_y : Real;
     size_gx : Real; n_x : Integer;  n_y : Integer; n_gx : Integer;
      b1 : Boolean; b2 : Boolean; b3 : Boolean; b4 : Boolean; b5 : Boolean; b6 : Boolean; b7 : Boolean; b8 : Boolean; b9 : Boolean;
      d1: Real; d2: Real; d3: Real; d4: Real; d5: Real; d6: Real; d7: Real; d8: Real; d9: Real;
      k1: Real; k2: Real; k3: Real; k4: Real; k5: Real; k6: Real; k7: Real; k8 : Real; k9 : Real;
      rhoCp1 : Real;  rhoCp2 : Real; rhoCp3 : Real;
      rhoCp4 : Real;  rhoCp5 : Real; rhoCp6 : Real;
      rhoCp7 : Real;  rhoCp8 : Real; rhoCp9 : Real;
      mplane1 : Real;   mplane2 : Real; mplane3 : Real;
      mplane4 : Real;   mplane5 : Real; mplane6 : Real;
      mplane7 : Real;   mplane8 : Real; mplane9 : Real;
      mortogonal1 : Real; mortogonal2 : Real; mortogonal3 : Real;
      mortogonal4 : Real; mortogonal5 : Real; mortogonal6 : Real;
      mortogonal7 : Real; mortogonal8 : Real; mortogonal9 : Real;
      alpha1 : Real;   alpha2 : Real;   alpha3 : Real;
      alpha4 : Real;   alpha5 : Real;   alpha6 : Real;
      alpha7 : Real;   alpha8 : Real;   alpha9 : Real;
      var time : Integer; Tamb : Real; Pdiss : Real;
      export3D : Boolean; exportxy2D : Boolean; exportx1D : Boolean;
      bfloat : Boolean); external 'Fourier.dll';
{$R *.dfm}

procedure TFormTopology.Button1Click(Sender: TObject);
var
   Rt : Real;
   d_check : Single;
   size_x, size_y, size_gx, distance_x, distance_y, Tamb, Pdiss : Real;
   n_x, n_y, n_gx, time, im, is1, i, j : Integer;
   bOk, bfloat : Boolean;
   f : TStrings; // переменная типа объект TStringList
   s : string; // записываемая строка и имя записываемого файла
   ShellInfo : TShellExecuteInfo;
   ExitCode : DWORD;
   QuoteParams : Boolean;

begin
   Tamb:=22.0;
   Pdiss:=1.0;
   if (FormLauncher.RadioGroup1.ItemIndex=0) then
   begin
      bfloat:=true;
   end
   else
   begin
     bfloat:=false;
   end;

   bOk:=true;
   LabelThermalresistance.Caption:='';

    if TryStrToFloat(FormConditions.Edit1.Text,d_check) then
   begin
      if (d_check>0.0) then
      begin
         Tamb:=StrToFloat(FormConditions.Edit1.Text);
      end
      else
      begin
          bOk:=false;
          FormConditions.Edit1.Color:=clRed;
      end;
   end
   else
   begin
     bOk:=false;
     FormConditions.Edit1.Color:=clRed;
   end;

    if TryStrToFloat(FormConditions.Edit2.Text,d_check) then
   begin
      if (d_check>0.0) then
      begin
         Pdiss:=StrToFloat(FormConditions.Edit2.Text);
      end
      else
      begin
          bOk:=false;
          FormConditions.Edit2.Color:=clRed;
      end;
   end
   else
   begin
     bOk:=false;
     FormConditions.Edit2.Color:=clRed;
   end;

   if TryStrToFloat(Edit1.Text,d_check) then
   begin
      if (d_check>0.0) then
      begin
         size_x:=StrToFloat(Edit1.Text);
      end
      else
      begin
          bOk:=false;
          Edit1.Color:=clRed;
      end;
   end
   else
   begin
     bOk:=false;
     Edit1.Color:=clRed;
   end;

    if TryStrToFloat(Edit4.Text,d_check) then
   begin
      if (d_check>=0.0) then
      begin
         size_gx:=StrToFloat(Edit4.Text);
      end
      else
      begin
          bOk:=false;
          Edit4.Color:=clRed;
      end;
   end
   else
   begin
     bOk:=false;
     Edit4.Color:=clRed;
   end;


    if TryStrToFloat(Edit3.Text,d_check) then
   begin
      if (d_check>0.0) then
      begin
         size_y:=StrToFloat(Edit3.Text);
      end
      else
      begin
          bOk:=false;
          Edit3.Color:=clRed;
      end;
   end
   else
   begin
     bOk:=false;
     Edit3.Color:=clRed;
   end;

    if TryStrToFloat(Edit2.Text,d_check) then
   begin
      if (d_check>=0.0) then
      begin
         distance_x:=StrToFloat(Edit2.Text);
      end
      else
      begin
          bOk:=false;
          Edit2.Color:=clRed;
      end;
   end
   else
   begin
     bOk:=false;
     Edit2.Color:=clRed;
   end;

    if TryStrToFloat(Edit5.Text,d_check) then
   begin
      if (d_check>=0.0) then
      begin
         distance_y:=StrToFloat(Edit5.Text);
      end
      else
      begin
          bOk:=false;
          Edit5.Color:=clRed;
      end;
   end
   else
   begin
     bOk:=false;
     Edit5.Color:=clRed;
   end;

   n_x:=ComboBox1.ItemIndex+1;
   n_y:=ComboBox3.ItemIndex+1;
   n_gx:=ComboBox2.ItemIndex+1;

   //Labelwait.Caption:='Please wait...';
  // sleep(1000);
  if (bOk and FormLayers.bOk_layers) then
  begin
   if (FormLauncher.RadioGroup1.ItemIndex=2) then
   begin

     // GPU Computing  6.06.2021

     Edit1.Color:=clWhite;
     Edit2.Color:=clWhite;
     Edit3.Color:=clWhite;
     Edit4.Color:=clWhite;
     Edit5.Color:=clWhite;
     FormConditions.Edit1.Color:=clWhite;
     FormConditions.Edit2.Color:=clWhite;
     time:=0;

     f:=TStringList.Create();

     s:=FloatToStr(Rt);
     f.Add(s);
     s:=FloatToStr(1.0e-6*size_x);
     f.Add(s);
     s:=FloatToStr(1.0e-6*size_y);
     f.Add(s);
     s:=FloatToStr(1.0e-6*distance_x);
     f.Add(s);
     s:=FloatToStr(1.0e-6*distance_y);
     f.Add(s);
     s:=FloatToStr(1.0e-6*size_gx);
     f.Add(s);
     s:=IntToStr(n_x);
     f.Add(s);
     s:=IntToStr(n_y);
     f.Add(s);
     s:=IntToStr(n_gx);
     f.Add(s);
     if (FormLayers.Panel9.Visible) then
     begin
        s:=IntToStr(1);
     end
     else
     begin
        s:=IntToStr(0);
     end;
     f.Add(s);
     if (FormLayers.Panel8.Visible) then
     begin
        s:=IntToStr(1);
     end
     else
     begin
        s:=IntToStr(0);
     end;
     f.Add(s);
     if (FormLayers.Panel7.Visible) then
     begin
        s:=IntToStr(1);
     end
     else
     begin
        s:=IntToStr(0);
     end;
     f.Add(s);
     if (FormLayers.Panel6.Visible) then
     begin
        s:=IntToStr(1);
     end
     else
     begin
        s:=IntToStr(0);
     end;
     f.Add(s);
     if (FormLayers.Panel5.Visible) then
     begin
        s:=IntToStr(1);
     end
     else
     begin
        s:=IntToStr(0);
     end;
     f.Add(s);
     if (FormLayers.Panel4.Visible) then
     begin
        s:=IntToStr(1);
     end
     else
     begin
        s:=IntToStr(0);
     end;
     f.Add(s);
     if (FormLayers.Panel3.Visible) then
     begin
        s:=IntToStr(1);
     end
     else
     begin
        s:=IntToStr(0);
     end;
     f.Add(s);
     if (FormLayers.Panel2.Visible) then
     begin
        s:=IntToStr(1);
     end
     else
     begin
        s:=IntToStr(0);
     end;
     f.Add(s);
     if (FormLayers.Panel1.Visible) then
     begin
        s:=IntToStr(1);
     end
     else
     begin
        s:=IntToStr(0);
     end;
     f.Add(s);

     s:=FloatToStr(1.0e-6*FormLayers.d1);
     f.Add(s);
     s:=FloatToStr(1.0e-6*FormLayers.d2);
     f.Add(s);
     s:=FloatToStr(1.0e-6*FormLayers.d3);
     f.Add(s);
     s:=FloatToStr(1.0e-6*FormLayers.d4);
     f.Add(s);
     s:=FloatToStr(1.0e-6*FormLayers.d5);
     f.Add(s);
     s:=FloatToStr(1.0e-6*FormLayers.d6);
     f.Add(s);
     s:=FloatToStr(1.0e-6*FormLayers.d7);
     f.Add(s);
     s:=FloatToStr(1.0e-6*FormLayers.d8);
     f.Add(s);
     s:=FloatToStr(1.0e-6*FormLayers.d9);
     f.Add(s);

     s:=FloatToStr(FormLayers.k1);
     f.Add(s);
     s:=FloatToStr(FormLayers.k2);
     f.Add(s);
     s:=FloatToStr(FormLayers.k3);
     f.Add(s);
     s:=FloatToStr(FormLayers.k4);
     f.Add(s);
     s:=FloatToStr(FormLayers.k5);
     f.Add(s);
     s:=FloatToStr(FormLayers.k6);
     f.Add(s);
     s:=FloatToStr(FormLayers.k7);
     f.Add(s);
     s:=FloatToStr(FormLayers.k8);
     f.Add(s);
     s:=FloatToStr(FormLayers.k9);
     f.Add(s);

      s:=FloatToStr(FormLayers.rhoCp1);
     f.Add(s);
     s:=FloatToStr(FormLayers.rhoCp2);
     f.Add(s);
     s:=FloatToStr(FormLayers.rhoCp3);
     f.Add(s);
     s:=FloatToStr(FormLayers.rhoCp4);
     f.Add(s);
     s:=FloatToStr(FormLayers.rhoCp5);
     f.Add(s);
     s:=FloatToStr(FormLayers.rhoCp6);
     f.Add(s);
     s:=FloatToStr(FormLayers.rhoCp7);
     f.Add(s);
     s:=FloatToStr(FormLayers.rhoCp8);
     f.Add(s);
     s:=FloatToStr(FormLayers.rhoCp9);
     f.Add(s);

     s:=FloatToStr(FormLayers.mplane1);
     f.Add(s);
     s:=FloatToStr(FormLayers.mplane2);
     f.Add(s);
     s:=FloatToStr(FormLayers.mplane3);
     f.Add(s);
     s:=FloatToStr(FormLayers.mplane4);
     f.Add(s);
     s:=FloatToStr(FormLayers.mplane5);
     f.Add(s);
     s:=FloatToStr(FormLayers.mplane6);
     f.Add(s);
     s:=FloatToStr(FormLayers.mplane7);
     f.Add(s);
     s:=FloatToStr(FormLayers.mplane8);
     f.Add(s);
     s:=FloatToStr(FormLayers.mplane9);
     f.Add(s);

     s:=FloatToStr(FormLayers.mortogonal1);
     f.Add(s);
     s:=FloatToStr(FormLayers.mortogonal2);
     f.Add(s);
     s:=FloatToStr(FormLayers.mortogonal3);
     f.Add(s);
     s:=FloatToStr(FormLayers.mortogonal4);
     f.Add(s);
     s:=FloatToStr(FormLayers.mortogonal5);
     f.Add(s);
     s:=FloatToStr(FormLayers.mortogonal6);
     f.Add(s);
     s:=FloatToStr(FormLayers.mortogonal7);
     f.Add(s);
     s:=FloatToStr(FormLayers.mortogonal8);
     f.Add(s);
     s:=FloatToStr(FormLayers.mortogonal9);
     f.Add(s);

     s:=FloatToStr(FormLayers.alpha1);
     f.Add(s);
     s:=FloatToStr(FormLayers.alpha2);
     f.Add(s);
     s:=FloatToStr(FormLayers.alpha3);
     f.Add(s);
     s:=FloatToStr(FormLayers.alpha4);
     f.Add(s);
     s:=FloatToStr(FormLayers.alpha5);
     f.Add(s);
     s:=FloatToStr(FormLayers.alpha6);
     f.Add(s);
     s:=FloatToStr(FormLayers.alpha7);
     f.Add(s);
     s:=FloatToStr(FormLayers.alpha8);
     f.Add(s);
     s:=FloatToStr(FormLayers.alpha9);
     f.Add(s);

     s:=FloatToStr(time);
     f.Add(s);
     s:=FloatToStr(Tamb);
     f.Add(s);
     s:=FloatToStr(Pdiss);
     f.Add(s);

     if (FormExport.CheckBox3D.Checked) then
     begin
        s:=IntToStr(1);
     end
     else
     begin
        s:=IntToStr(0);
     end;
     f.Add(s);

     if (FormExport.CheckBox2Dexport.Checked) then
     begin
        s:=IntToStr(1);
     end
     else
     begin
        s:=IntToStr(0);
     end;
     f.Add(s);

     if (FormExport.CheckBox1Dexport.Checked) then
     begin
        s:=IntToStr(1);
     end
     else
     begin
        s:=IntToStr(0);
     end;
     f.Add(s);

     if (bfloat) then
     begin
        s:=IntToStr(1);
     end
     else
     begin
        s:=IntToStr(0);
     end;
     f.Add(s);

     // GPU id
     s:=IntToStr(FormLauncher.ComboBoxGPUid.ItemIndex);
     f.Add(s);

     // 25.07.2021
     // Замена запятых на точку так чтобы нормально считывалось в Visual Studio.
     for i := 0 to f.Count-1 do
     begin
       s:=f.Strings[i];
       for j := 1 to length(s) do
       begin
          if (s[j]=',') then
          begin
             s[j]:='.';
          end;
       end;
       f.Strings[i]:=s;
     end;


     f.SaveToFile('source.txt');  // сохраняю результат
     f.Free;

        ShellInfo.cbSize:=SizeOf(ShellInfo);
        ShellInfo.fMask:=SEE_MASK_NOCLOSEPROCESS;
        ShellInfo.Wnd:=HWND_DESKTOP;
        ShellInfo.lpVerb:='open';
        ShellInfo.lpFile:=PChar('"Fourier_GPU.exe"');
        QuoteParams:=true;
        if QuoteParams then
        ShellInfo.lpParameters:=PChar('""')
        else
        ShellInfo.lpParameters:=PChar('');
        ShellInfo.lpDirectory:=PChar('".\"');
        ShellInfo.nShow:=SW_SHOWNORMAL;
        if not ShellExecuteEx(@ShellInfo) then
        RaiseLastOSError;
        if ShellInfo.hProcess<>0 then
        try
          WaitForSingleObjectEx(ShellInfo.hProcess,INFINITE,false);
          GetExitCodeProcess(ShellInfo.hProcess,ExitCode);
        finally
           CloseHandle(ShellInfo.hProcess);
        end;

   end
   else
   begin
     Edit1.Color:=clWhite;
     Edit2.Color:=clWhite;
     Edit3.Color:=clWhite;
     Edit4.Color:=clWhite;
     Edit5.Color:=clWhite;
     FormConditions.Edit1.Color:=clWhite;
     FormConditions.Edit2.Color:=clWhite;
     time:=0;
     fourier_solve(Rt,1.0e-6*size_x, 1.0e-6*size_y, 1.0e-6*distance_x, 1.0e-6*distance_y,
      1.0e-6*size_gx, n_x,  n_y, n_gx,
      FormLayers.Panel9.Visible, FormLayers.Panel8.Visible, FormLayers.Panel7.Visible,
      FormLayers.Panel6.Visible, FormLayers.Panel5.Visible, FormLayers.Panel4.Visible,
      FormLayers.Panel3.Visible, FormLayers.Panel2.Visible, FormLayers.Panel1.Visible,
      1.0e-6*FormLayers.d1, 1.0e-6*FormLayers.d2, 1.0e-6*FormLayers.d3, 1.0e-6*FormLayers.d4,
      1.0e-6*FormLayers.d5, 1.0e-6*FormLayers.d6,
      1.0e-6*FormLayers.d7, 1.0e-6*FormLayers.d8, 1.0e-6*FormLayers.d9,
      FormLayers.k1, FormLayers.k2, FormLayers.k3, FormLayers.k4, FormLayers.k5, FormLayers.k6,
      FormLayers.k7, FormLayers.k8, FormLayers.k9 ,
      FormLayers.rhoCp1, FormLayers.rhoCp2, FormLayers.rhoCp3,
      FormLayers.rhoCp4, FormLayers.rhoCp5, FormLayers.rhoCp6,
      FormLayers.rhoCp7, FormLayers.rhoCp8, FormLayers.rhoCp9,
      FormLayers.mplane1, FormLayers.mplane2, FormLayers.mplane3,
      FormLayers.mplane4, FormLayers.mplane5, FormLayers.mplane6,
      FormLayers.mplane7, FormLayers.mplane8, FormLayers.mplane9,
      FormLayers.mortogonal1, FormLayers.mortogonal2, FormLayers.mortogonal3,
      FormLayers.mortogonal4, FormLayers.mortogonal5, FormLayers.mortogonal6,
      FormLayers.mortogonal7, FormLayers.mortogonal8, FormLayers.mortogonal9,
      FormLayers.alpha1,  FormLayers.alpha2, FormLayers.alpha3,
      FormLayers.alpha4,  FormLayers.alpha5, FormLayers.alpha6,
      FormLayers.alpha7,  FormLayers.alpha8, FormLayers.alpha9,
       time, Tamb, Pdiss,
      FormExport.CheckBox3D.Checked, FormExport.CheckBox2Dexport.Checked,
      FormExport.CheckBox1Dexport.Checked, bfloat);
     LabelThermalresistance.Caption:=FormatFloat('##.##',Rt);
     im:=time div 60000;
     is1:=(time - 60000*im) div 1000;
     // Время решения.
     Labelwait.Caption:=IntToStr(im)+' m '+IntToStr(is1)+' s '+IntToStr(time - 60000*im - 1000*is1)+' ms ';
   end;
  end
  else
  begin
     Labelwait.Caption:='';
  end;
end;

// Задаёт стек слоёв.
procedure TFormTopology.Button2Click(Sender: TObject);
begin
   FormLayers.ShowModal;
end;

procedure TFormTopology.Close1Click(Sender: TObject);
begin
   Close;
end;

// Задаёт температуру корпуса и мощность тепловыделения.
procedure TFormTopology.Define2Click(Sender: TObject);
begin
   FormConditions.ShowModal;
end;

// Задаёт настройки записи в файл.
procedure TFormTopology.Export1Click(Sender: TObject);
begin
    FormExport.ShowModal;
end;

procedure TFormTopology.FormCreate(Sender: TObject);
begin

  bFirst_Matherial:=true;

   SetLength(matherial,41);

   matherial[0].name:='GaN';
   matherial[0].density:=6150;
   matherial[0].heatCapasity:=700;
   matherial[0].thermalConductivity:=130;
   matherial[0].multiplyerConductivityPlane:=1.0;
   matherial[0].multiplyerConductivityNormal:=1.0;
   matherial[0].alphaForTemperatureDepend:=-0.43;

   matherial[1].name:='Si';
   matherial[1].density:=2330;
   matherial[1].heatCapasity:=711;
   matherial[1].thermalConductivity:=148;
   matherial[1].multiplyerConductivityPlane:=1.0;
   matherial[1].multiplyerConductivityNormal:=1.0;
   matherial[1].alphaForTemperatureDepend:=-1.35;

   matherial[2].name:='SiC';
   matherial[2].density:=3210;
   matherial[2].heatCapasity:=690;
   matherial[2].thermalConductivity:=370;
   matherial[2].multiplyerConductivityPlane:=1.27;
   matherial[2].multiplyerConductivityNormal:=1.0;
   matherial[2].alphaForTemperatureDepend:=-1.5;

   matherial[3].name:='сапфир';
   matherial[3].density:=4000;
   matherial[3].heatCapasity:=718;
   matherial[3].thermalConductivity:=28;
   matherial[3].multiplyerConductivityPlane:=1.0;
   matherial[3].multiplyerConductivityNormal:=1.0;
   matherial[3].alphaForTemperatureDepend:=-1.0;

   matherial[4].name:='алмаз';
   matherial[4].density:=3500;
   matherial[4].heatCapasity:=520;
   matherial[4].thermalConductivity:=2000;
   matherial[4].multiplyerConductivityPlane:=1.0;
   matherial[4].multiplyerConductivityNormal:=1.0;
   matherial[4].alphaForTemperatureDepend:=-1.85;

   matherial[5].name:='GaAs';
   matherial[5].density:=5300;
   matherial[5].heatCapasity:=330;
   matherial[5].thermalConductivity:=47;
   matherial[5].multiplyerConductivityPlane:=1.0;
   matherial[5].multiplyerConductivityNormal:=1.0;
   matherial[5].alphaForTemperatureDepend:=-1.25;

   matherial[6].name:='поликор';
   matherial[6].density:=3760;
   matherial[6].heatCapasity:=750;
   matherial[6].thermalConductivity:=25;
   matherial[6].multiplyerConductivityPlane:=1.0;
   matherial[6].multiplyerConductivityNormal:=1.0;
   matherial[6].alphaForTemperatureDepend:=0.0;

   matherial[7].name:='AuSn';
   matherial[7].density:=14510;
   matherial[7].heatCapasity:=143;
   matherial[7].thermalConductivity:=57;
   matherial[7].multiplyerConductivityPlane:=1.0;
   matherial[7].multiplyerConductivityNormal:=1.0;
   matherial[7].alphaForTemperatureDepend:=0.0;

   matherial[8].name:='Cu';
   matherial[8].density:=8930;
   matherial[8].heatCapasity:=390;
   matherial[8].thermalConductivity:=390;
   matherial[8].multiplyerConductivityPlane:=1.0;
   matherial[8].multiplyerConductivityNormal:=1.0;
   matherial[8].alphaForTemperatureDepend:=0.0;

   matherial[9].name:='МД40';
   matherial[9].density:=8900;
   matherial[9].heatCapasity:=318;
   matherial[9].thermalConductivity:=210;
   matherial[9].multiplyerConductivityPlane:=1.0;
   matherial[9].multiplyerConductivityNormal:=1.0;
   matherial[9].alphaForTemperatureDepend:=0.0;

   matherial[10].name:='Клей ЭЧЭС';
   matherial[10].density:=1000;
   matherial[10].heatCapasity:=100;
   matherial[10].thermalConductivity:=4;
   matherial[10].multiplyerConductivityPlane:=1.0;
   matherial[10].multiplyerConductivityNormal:=1.0;
   matherial[10].alphaForTemperatureDepend:=0.0;

   matherial[11].name:='gold';
   matherial[11].density:=19300;
   matherial[11].heatCapasity:=126;
   matherial[11].thermalConductivity:=293;
   matherial[11].multiplyerConductivityPlane:=1.0;
   matherial[11].multiplyerConductivityNormal:=1.0;
   matherial[11].alphaForTemperatureDepend:=0.0;

   matherial[12].name:='Aluminium';
   matherial[12].density:=2700;
   matherial[12].heatCapasity:=920;
   matherial[12].thermalConductivity:=201;
   matherial[12].multiplyerConductivityPlane:=1.0;
   matherial[12].multiplyerConductivityNormal:=1.0;
   matherial[12].alphaForTemperatureDepend:=0.0;

   matherial[13].name:='Д16';
   matherial[13].density:=2800;
   matherial[13].heatCapasity:=921;
   matherial[13].thermalConductivity:=164;
   matherial[13].multiplyerConductivityPlane:=1.0;
   matherial[13].multiplyerConductivityNormal:=1.0;
   matherial[13].alphaForTemperatureDepend:=0.0;

   matherial[14].name:='Латунь';
   matherial[14].density:=8500;
   matherial[14].heatCapasity:=377;
   matherial[14].thermalConductivity:=109;
   matherial[14].multiplyerConductivityPlane:=1.0;
   matherial[14].multiplyerConductivityNormal:=1.0;
   matherial[14].alphaForTemperatureDepend:=0.0;

   matherial[15].name:='стекло';
   matherial[15].density:=2500;
   matherial[15].heatCapasity:=753;
   matherial[15].thermalConductivity:=0.85;
   matherial[15].multiplyerConductivityPlane:=1.0;
   matherial[15].multiplyerConductivityNormal:=1.0;
   matherial[15].alphaForTemperatureDepend:=0.0;

   matherial[16].name:='резина';
   matherial[16].density:=1100;
   matherial[16].heatCapasity:=2005;
   matherial[16].thermalConductivity:=0.16;
   matherial[16].multiplyerConductivityPlane:=1.0;
   matherial[16].multiplyerConductivityNormal:=1.0;
   matherial[16].alphaForTemperatureDepend:=0.0;

   matherial[17].name:='AuSi';
   matherial[17].density:=19300;
   matherial[17].heatCapasity:=125;
   matherial[17].thermalConductivity:=190;
   matherial[17].multiplyerConductivityPlane:=1.0;
   matherial[17].multiplyerConductivityNormal:=1.0;
   matherial[17].alphaForTemperatureDepend:=0.0;

   matherial[18].name:='фторопласт-3';
   matherial[18].density:=2150;
   matherial[18].heatCapasity:=1046;
   matherial[18].thermalConductivity:=0.058;
   matherial[18].multiplyerConductivityPlane:=1.0;
   matherial[18].multiplyerConductivityNormal:=1.0;
   matherial[18].alphaForTemperatureDepend:=0.0;

   matherial[19].name:='фторопласт-4';
   matherial[19].density:=2150;
   matherial[19].heatCapasity:=1046;
   matherial[19].thermalConductivity:=0.233;
   matherial[19].multiplyerConductivityPlane:=1.0;
   matherial[19].multiplyerConductivityNormal:=1.0;
   matherial[19].alphaForTemperatureDepend:=0.0;

   matherial[20].name:='Titan';
   matherial[20].density:=4540;
   matherial[20].heatCapasity:=532;
   matherial[20].thermalConductivity:=15;
   matherial[20].multiplyerConductivityPlane:=1.0;
   matherial[20].multiplyerConductivityNormal:=1.0;
   matherial[20].alphaForTemperatureDepend:=0.0;

   matherial[21].name:='сталь';
   matherial[21].density:=7860;
   matherial[21].heatCapasity:=460;
   matherial[21].thermalConductivity:=16;
   matherial[21].multiplyerConductivityPlane:=1.0;
   matherial[21].multiplyerConductivityNormal:=1.0;
   matherial[21].alphaForTemperatureDepend:=0.0;

   matherial[22].name:='Ковар';
   matherial[22].density:=8300;
   matherial[22].heatCapasity:=669;
   matherial[22].thermalConductivity:=19;
   matherial[22].multiplyerConductivityPlane:=1.0;
   matherial[22].multiplyerConductivityNormal:=1.0;
   matherial[22].alphaForTemperatureDepend:=0.0;

   matherial[23].name:='Ni';
   matherial[23].density:=8900;
   matherial[23].heatCapasity:=418;
   matherial[23].thermalConductivity:=85;
   matherial[23].multiplyerConductivityPlane:=1.0;
   matherial[23].multiplyerConductivityNormal:=1.0;
   matherial[23].alphaForTemperatureDepend:=0.0;

   matherial[24].name:='BeO';
   matherial[24].density:=2900;
   matherial[24].heatCapasity:=1200;
   matherial[24].thermalConductivity:=250;
   matherial[24].multiplyerConductivityPlane:=1.0;
   matherial[24].multiplyerConductivityNormal:=1.0;
   matherial[24].alphaForTemperatureDepend:=0.0;

   matherial[25].name:='air';
   matherial[25].density:=1.1614;
   matherial[25].heatCapasity:=1005;
   matherial[25].thermalConductivity:=0.025;
   matherial[25].multiplyerConductivityPlane:=1.0;
   matherial[25].multiplyerConductivityNormal:=1.0;
   matherial[25].alphaForTemperatureDepend:=0.0;

   matherial[26].name:='In';
   matherial[26].density:=7300;
   matherial[26].heatCapasity:=243;
   matherial[26].thermalConductivity:=34;
   matherial[26].multiplyerConductivityPlane:=1.0;
   matherial[26].multiplyerConductivityNormal:=1.0;
   matherial[26].alphaForTemperatureDepend:=0.0;

   matherial[27].name:='AuGe';
   matherial[27].density:=19300;
   matherial[27].heatCapasity:=126;
   matherial[27].thermalConductivity:=147;
   matherial[27].multiplyerConductivityPlane:=1.0;
   matherial[27].multiplyerConductivityNormal:=1.0;
   matherial[27].alphaForTemperatureDepend:=0.0;

   matherial[28].name:='КПТ-8';
   matherial[28].density:=2800;
   matherial[28].heatCapasity:=1000;
   matherial[28].thermalConductivity:=0.7;
   matherial[28].multiplyerConductivityPlane:=1.0;
   matherial[28].multiplyerConductivityNormal:=1.0;
   matherial[28].alphaForTemperatureDepend:=0.0;

   matherial[29].name:='AlN';
   matherial[29].density:=2500;
   matherial[29].heatCapasity:=896;
   matherial[29].thermalConductivity:=140;
   matherial[29].multiplyerConductivityPlane:=1.0;
   matherial[29].multiplyerConductivityNormal:=1.0;
   matherial[29].alphaForTemperatureDepend:=0.0;

   matherial[30].name:='130-AlN';
   matherial[30].density:=2500;
   matherial[30].heatCapasity:=896;
   matherial[30].thermalConductivity:=130;
   matherial[30].multiplyerConductivityPlane:=1.0;
   matherial[30].multiplyerConductivityNormal:=1.0;
   matherial[30].alphaForTemperatureDepend:=0.0;

   matherial[31].name:='150-AlN';
   matherial[31].density:=2500;
   matherial[31].heatCapasity:=896;
   matherial[31].thermalConductivity:=150;
   matherial[31].multiplyerConductivityPlane:=1.0;
   matherial[31].multiplyerConductivityNormal:=1.0;
   matherial[31].alphaForTemperatureDepend:=0.0;

   matherial[32].name:='200-AlN';
   matherial[32].density:=2500;
   matherial[32].heatCapasity:=896;
   matherial[32].thermalConductivity:=200;
   matherial[32].multiplyerConductivityPlane:=1.0;
   matherial[32].multiplyerConductivityNormal:=1.0;
   matherial[32].alphaForTemperatureDepend:=0.0;

   matherial[33].name:='CuW';
   matherial[33].density:=15150;
   matherial[33].heatCapasity:=174;
   matherial[33].thermalConductivity:=220;
   matherial[33].multiplyerConductivityPlane:=1.0;
   matherial[33].multiplyerConductivityNormal:=1.0;
   matherial[33].alphaForTemperatureDepend:=0.0;

   matherial[34].name:='CuW-75';
   matherial[34].density:=14800;
   matherial[34].heatCapasity:=190;
   matherial[34].thermalConductivity:=189;
   matherial[34].multiplyerConductivityPlane:=1.0;
   matherial[34].multiplyerConductivityNormal:=1.0;
   matherial[34].alphaForTemperatureDepend:=0.0;

   matherial[35].name:='22ХС';
   matherial[35].density:=3600;
   matherial[35].heatCapasity:=921;
   matherial[35].thermalConductivity:=32;
   matherial[35].multiplyerConductivityPlane:=1.0;
   matherial[35].multiplyerConductivityNormal:=1.0;
   matherial[35].alphaForTemperatureDepend:=0.0;

   matherial[36].name:='ПСР-72';
   matherial[36].density:=10500;
   matherial[36].heatCapasity:=234;
   matherial[36].thermalConductivity:=300;
   matherial[36].multiplyerConductivityPlane:=1.0;
   matherial[36].multiplyerConductivityNormal:=1.0;
   matherial[36].alphaForTemperatureDepend:=0.0;

   matherial[37].name:='ПОС-60';
   matherial[37].density:=9220;
   matherial[37].heatCapasity:=176;
   matherial[37].thermalConductivity:=50;
   matherial[37].multiplyerConductivityPlane:=1.0;
   matherial[37].multiplyerConductivityNormal:=1.0;
   matherial[37].alphaForTemperatureDepend:=0.0;

   matherial[38].name:='FR-4';
   matherial[38].density:=1900;
   matherial[38].heatCapasity:=1300;
   matherial[38].thermalConductivity:=0.39;
   matherial[38].multiplyerConductivityPlane:=1.0;
   matherial[38].multiplyerConductivityNormal:=1.0;
   matherial[38].alphaForTemperatureDepend:=0.0;

   matherial[39].name:='канифоль';
   matherial[39].density:=1080;
   matherial[39].heatCapasity:=1340;
   matherial[39].thermalConductivity:=0.164;
   matherial[39].multiplyerConductivityPlane:=1.0;
   matherial[39].multiplyerConductivityNormal:=1.0;
   matherial[39].alphaForTemperatureDepend:=0.0;

   matherial[40].name:='Arlon-AD1000';
   matherial[40].density:=0;
   matherial[40].heatCapasity:=0;
   matherial[40].thermalConductivity:=0.81;
   matherial[40].multiplyerConductivityPlane:=1.0;
   matherial[40].multiplyerConductivityNormal:=1.0;
   matherial[40].alphaForTemperatureDepend:=0.0;

end;

procedure TFormTopology.Launcher1Click(Sender: TObject);
begin
    FormLauncher.ShowModal;
end;

procedure TFormTopology.Matherials1Click(Sender: TObject);
begin
   if (bFirst_Matherial) then
   begin
     bFirst_Matherial:=false;
     FormMatherials.ComboBox1.ItemIndex:=0;
     FormMatherials.ComboBox1Click(Sender);
   end;

   FormMatherials.ShowModal;
end;

end.
