unit UnitLayers2;

interface

uses
  Winapi.Windows, Winapi.Messages, System.SysUtils, System.Variants, System.Classes, Vcl.Graphics,
  Vcl.Controls, Vcl.Forms, Vcl.Dialogs, Vcl.StdCtrls, Vcl.ExtCtrls;

type
  TFormLayers = class(TForm)
    Label1: TLabel;
    ComboBox1: TComboBox;
    Panel1: TPanel;
    Label2: TLabel;
    Label3: TLabel;
    ComboBox2: TComboBox;
    Label4: TLabel;
    Edit1: TEdit;
    Label5: TLabel;
    Panel2: TPanel;
    Label6: TLabel;
    Label7: TLabel;
    ComboBox3: TComboBox;
    Edit2: TEdit;
    Panel3: TPanel;
    Label8: TLabel;
    Label9: TLabel;
    ComboBox4: TComboBox;
    Edit3: TEdit;
    Panel4: TPanel;
    Label10: TLabel;
    Label11: TLabel;
    ComboBox5: TComboBox;
    Edit4: TEdit;
    Panel5: TPanel;
    Label12: TLabel;
    Label13: TLabel;
    ComboBox6: TComboBox;
    Edit5: TEdit;
    Panel6: TPanel;
    Label14: TLabel;
    Label15: TLabel;
    ComboBox7: TComboBox;
    Edit6: TEdit;
    Panel7: TPanel;
    Label16: TLabel;
    Label17: TLabel;
    ComboBox8: TComboBox;
    Edit7: TEdit;
    Panel8: TPanel;
    Label18: TLabel;
    Label19: TLabel;
    ComboBox9: TComboBox;
    Edit8: TEdit;
    Panel9: TPanel;
    Label20: TLabel;
    Label21: TLabel;
    ComboBox10: TComboBox;
    Edit9: TEdit;
    Label22: TLabel;
    Button1: TButton;
    procedure ComboBox1Change(Sender: TObject);
    procedure FormClose(Sender: TObject; var Action: TCloseAction);
    procedure Button1Click(Sender: TObject);
    procedure FormCreate(Sender: TObject);
  private
    { Private declarations }
  public
    { Public declarations }
    d1, d2, d3, d4, d5, d6, d7, d8, d9 : Real;
    k1, k2, k3, k4, k5, k6, k7, k8, k9 : Real;
    alpha1, alpha2, alpha3,  alpha4, alpha5, alpha6, alpha7, alpha8, alpha9 : Real;
    s1, s2, s3, s4, s5, s6, s7, s8, s9 : string;
    rhoCp1, rhoCp2, rhoCp3,   rhoCp4, rhoCp5, rhoCp6,  rhoCp7, rhoCp8, rhoCp9 : Real;
    mplane1, mplane2, mplane3, mplane4, mplane5, mplane6, mplane7, mplane8, mplane9 : Real;
    mortogonal1, mortogonal2, mortogonal3, mortogonal4, mortogonal5, mortogonal6, mortogonal7, mortogonal8, mortogonal9 : Real;
    bOk_layers : Boolean;

    procedure LoadK(Sender: TObject);
  end;

var
  FormLayers: TFormLayers;

implementation

{$R *.dfm}

uses Main1;

procedure TFormLayers.LoadK(Sender: TObject);
begin

    s1:= FormTopology.matherial[ComboBox10.ItemIndex].name;
    s2:= FormTopology.matherial[ComboBox9.ItemIndex].name;
    s3:= FormTopology.matherial[ComboBox8.ItemIndex].name;
    s4:= FormTopology.matherial[ComboBox7.ItemIndex].name;
    s5:= FormTopology.matherial[ComboBox6.ItemIndex].name;
    s6:= FormTopology.matherial[ComboBox5.ItemIndex].name;
    s7:= FormTopology.matherial[ComboBox4.ItemIndex].name;
    s8:= FormTopology.matherial[ComboBox3.ItemIndex].name;
    s9:= FormTopology.matherial[ComboBox2.ItemIndex].name;

    alpha1:= FormTopology.matherial[ComboBox10.ItemIndex].alphaForTemperatureDepend;
    alpha2:= FormTopology.matherial[ComboBox9.ItemIndex].alphaForTemperatureDepend;
    alpha3:= FormTopology.matherial[ComboBox8.ItemIndex].alphaForTemperatureDepend;
    alpha4:= FormTopology.matherial[ComboBox7.ItemIndex].alphaForTemperatureDepend;
    alpha5:= FormTopology.matherial[ComboBox6.ItemIndex].alphaForTemperatureDepend;
    alpha6:= FormTopology.matherial[ComboBox5.ItemIndex].alphaForTemperatureDepend;
    alpha7:= FormTopology.matherial[ComboBox4.ItemIndex].alphaForTemperatureDepend;
    alpha8:= FormTopology.matherial[ComboBox3.ItemIndex].alphaForTemperatureDepend;
    alpha9:= FormTopology.matherial[ComboBox2.ItemIndex].alphaForTemperatureDepend;


    k1:= FormTopology.matherial[ComboBox10.ItemIndex].thermalConductivity;
    k2:= FormTopology.matherial[ComboBox9.ItemIndex].thermalConductivity;
    k3:= FormTopology.matherial[ComboBox8.ItemIndex].thermalConductivity;
    k4:= FormTopology.matherial[ComboBox7.ItemIndex].thermalConductivity;
    k5:= FormTopology.matherial[ComboBox6.ItemIndex].thermalConductivity;
    k6:= FormTopology.matherial[ComboBox5.ItemIndex].thermalConductivity;
    k7:= FormTopology.matherial[ComboBox4.ItemIndex].thermalConductivity;
    k8:= FormTopology.matherial[ComboBox3.ItemIndex].thermalConductivity;
    k9:= FormTopology.matherial[ComboBox2.ItemIndex].thermalConductivity;

    rhoCp1:= FormTopology.matherial[ComboBox10.ItemIndex].density*FormTopology.matherial[ComboBox10.ItemIndex].heatCapasity;
    rhoCp2:= FormTopology.matherial[ComboBox9.ItemIndex].density*FormTopology.matherial[ComboBox9.ItemIndex].heatCapasity;
    rhoCp3:= FormTopology.matherial[ComboBox8.ItemIndex].density*FormTopology.matherial[ComboBox9.ItemIndex].heatCapasity;
    rhoCp4:= FormTopology.matherial[ComboBox7.ItemIndex].density*FormTopology.matherial[ComboBox9.ItemIndex].heatCapasity;
    rhoCp5:= FormTopology.matherial[ComboBox6.ItemIndex].density*FormTopology.matherial[ComboBox9.ItemIndex].heatCapasity;
    rhoCp6:= FormTopology.matherial[ComboBox5.ItemIndex].density*FormTopology.matherial[ComboBox9.ItemIndex].heatCapasity;
    rhoCp7:= FormTopology.matherial[ComboBox4.ItemIndex].density*FormTopology.matherial[ComboBox9.ItemIndex].heatCapasity;
    rhoCp8:= FormTopology.matherial[ComboBox3.ItemIndex].density*FormTopology.matherial[ComboBox9.ItemIndex].heatCapasity;
    rhoCp9:= FormTopology.matherial[ComboBox2.ItemIndex].density*FormTopology.matherial[ComboBox9.ItemIndex].heatCapasity;

    mplane1:= FormTopology.matherial[ComboBox10.ItemIndex].multiplyerConductivityPlane;
    mplane2:= FormTopology.matherial[ComboBox9.ItemIndex].multiplyerConductivityPlane;
    mplane3:= FormTopology.matherial[ComboBox8.ItemIndex].multiplyerConductivityPlane;
    mplane4:= FormTopology.matherial[ComboBox7.ItemIndex].multiplyerConductivityPlane;
    mplane5:= FormTopology.matherial[ComboBox6.ItemIndex].multiplyerConductivityPlane;
    mplane6:= FormTopology.matherial[ComboBox5.ItemIndex].multiplyerConductivityPlane;
    mplane7:= FormTopology.matherial[ComboBox4.ItemIndex].multiplyerConductivityPlane;
    mplane8:= FormTopology.matherial[ComboBox3.ItemIndex].multiplyerConductivityPlane;
    mplane9:= FormTopology.matherial[ComboBox2.ItemIndex].multiplyerConductivityPlane;

    mortogonal1:= FormTopology.matherial[ComboBox10.ItemIndex].multiplyerConductivityNormal;
    mortogonal2:= FormTopology.matherial[ComboBox9.ItemIndex].multiplyerConductivityNormal;
    mortogonal3:= FormTopology.matherial[ComboBox8.ItemIndex].multiplyerConductivityNormal;
    mortogonal4:= FormTopology.matherial[ComboBox7.ItemIndex].multiplyerConductivityNormal;
    mortogonal5:= FormTopology.matherial[ComboBox6.ItemIndex].multiplyerConductivityNormal;
    mortogonal6:= FormTopology.matherial[ComboBox5.ItemIndex].multiplyerConductivityNormal;
    mortogonal7:= FormTopology.matherial[ComboBox4.ItemIndex].multiplyerConductivityNormal;
    mortogonal8:= FormTopology.matherial[ComboBox3.ItemIndex].multiplyerConductivityNormal;
    mortogonal9:= FormTopology.matherial[ComboBox2.ItemIndex].multiplyerConductivityNormal;



end;

procedure TFormLayers.Button1Click(Sender: TObject);
begin
   Close;
end;

procedure TFormLayers.ComboBox1Change(Sender: TObject);
begin
   case ComboBox1.ItemIndex of
       0 : begin
         Panel1.Visible:=false;
         Panel2.Visible:=false;
         Panel3.Visible:=false;
         Panel4.Visible:=false;
         Panel5.Visible:=false;
         Panel6.Visible:=false;
         Panel7.Visible:=false;
         Panel8.Visible:=false;
         Panel9.Visible:=true;
       end;
        1 : begin
         Panel1.Visible:=false;
         Panel2.Visible:=false;
         Panel3.Visible:=false;
         Panel4.Visible:=false;
         Panel5.Visible:=false;
         Panel6.Visible:=false;
         Panel7.Visible:=false;
         Panel8.Visible:=true;
         Panel9.Visible:=true;
       end;
        2 : begin
         Panel1.Visible:=false;
         Panel2.Visible:=false;
         Panel3.Visible:=false;
         Panel4.Visible:=false;
         Panel5.Visible:=false;
         Panel6.Visible:=false;
         Panel7.Visible:=true;
         Panel8.Visible:=true;
         Panel9.Visible:=true;
       end;
        3 : begin
         Panel1.Visible:=false;
         Panel2.Visible:=false;
         Panel3.Visible:=false;
         Panel4.Visible:=false;
         Panel5.Visible:=false;
         Panel6.Visible:=true;
         Panel7.Visible:=true;
         Panel8.Visible:=true;
         Panel9.Visible:=true;
       end;
        4 : begin
         Panel1.Visible:=false;
         Panel2.Visible:=false;
         Panel3.Visible:=false;
         Panel4.Visible:=false;
         Panel5.Visible:=true;
         Panel6.Visible:=true;
         Panel7.Visible:=true;
         Panel8.Visible:=true;
         Panel9.Visible:=true;
       end;
        5 : begin
         Panel1.Visible:=false;
         Panel2.Visible:=false;
         Panel3.Visible:=false;
         Panel4.Visible:=true;
         Panel5.Visible:=true;
         Panel6.Visible:=true;
         Panel7.Visible:=true;
         Panel8.Visible:=true;
         Panel9.Visible:=true;
       end;
        6 : begin
         Panel1.Visible:=false;
         Panel2.Visible:=false;
         Panel3.Visible:=true;
         Panel4.Visible:=true;
         Panel5.Visible:=true;
         Panel6.Visible:=true;
         Panel7.Visible:=true;
         Panel8.Visible:=true;
         Panel9.Visible:=true;
       end;
        7 : begin
         Panel1.Visible:=false;
         Panel2.Visible:=true;
         Panel3.Visible:=true;
         Panel4.Visible:=true;
         Panel5.Visible:=true;
         Panel6.Visible:=true;
         Panel7.Visible:=true;
         Panel8.Visible:=true;
         Panel9.Visible:=true;
       end;
        8 : begin
         Panel1.Visible:=true;
         Panel2.Visible:=true;
         Panel3.Visible:=true;
         Panel4.Visible:=true;
         Panel5.Visible:=true;
         Panel6.Visible:=true;
         Panel7.Visible:=true;
         Panel8.Visible:=true;
         Panel9.Visible:=true;
       end;
   end;
end;



procedure TFormLayers.FormClose(Sender: TObject; var Action: TCloseAction);
var
  d_check : Single;
  bOk : Boolean;
begin

   bOk:=true;

   Edit1.Color:=clWhite;
   Edit2.Color:=clWhite;
   Edit3.Color:=clWhite;
   Edit4.Color:=clWhite;
   Edit5.Color:=clWhite;
   Edit6.Color:=clWhite;
   Edit7.Color:=clWhite;
   Edit8.Color:=clWhite;
   Edit9.Color:=clWhite;

   if (Panel9.Visible) then
   begin
     if TryStrToFloat(Edit9.Text,d_check) then
     begin
        if (d_check>0.0) then
        begin
           d1:=StrToFloat(Edit9.Text);
        end
         else
        begin
           bOk:=false;
           Edit9.Color:=clRed;
        end;
     end
      else
     begin
        bOk:=false;
        Edit9.Color:=clRed;
     end;
   end;

    if (Panel8.Visible) then
   begin
     if TryStrToFloat(Edit8.Text,d_check) then
     begin
        if (d_check>0.0) then
        begin
           d2:=StrToFloat(Edit8.Text);
        end
         else
        begin
           bOk:=false;
           Edit8.Color:=clRed;
        end;
     end
      else
     begin
        bOk:=false;
        Edit8.Color:=clRed;
     end;
   end;

    if (Panel7.Visible) then
   begin
     if TryStrToFloat(Edit7.Text,d_check) then
     begin
        if (d_check>0.0) then
        begin
           d3:=StrToFloat(Edit7.Text);
        end
         else
        begin
           bOk:=false;
           Edit7.Color:=clRed;
        end;
     end
      else
     begin
        bOk:=false;
        Edit7.Color:=clRed;
     end;
   end;

    if (Panel6.Visible) then
   begin
     if TryStrToFloat(Edit6.Text,d_check) then
     begin
        if (d_check>0.0) then
        begin
           d4:=StrToFloat(Edit6.Text);
        end
         else
        begin
           bOk:=false;
           Edit6.Color:=clRed;
        end;
     end
      else
     begin
        bOk:=false;
        Edit6.Color:=clRed;
     end;
   end;

    if (Panel5.Visible) then
   begin
     if TryStrToFloat(Edit5.Text,d_check) then
     begin
        if (d_check>0.0) then
        begin
           d5:=StrToFloat(Edit5.Text);
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
   end;

    if (Panel4.Visible) then
   begin
     if TryStrToFloat(Edit4.Text,d_check) then
     begin
        if (d_check>0.0) then
        begin
           d6:=StrToFloat(Edit4.Text);
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
   end;

     if (Panel3.Visible) then
   begin
     if TryStrToFloat(Edit3.Text,d_check) then
     begin
        if (d_check>0.0) then
        begin
           d7:=StrToFloat(Edit3.Text);
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
   end;

     if (Panel2.Visible) then
   begin
     if TryStrToFloat(Edit2.Text,d_check) then
     begin
        if (d_check>0.0) then
        begin
           d8:=StrToFloat(Edit2.Text);
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
   end;

    if (Panel1.Visible) then
   begin
     if TryStrToFloat(Edit1.Text,d_check) then
     begin
        if (d_check>0.0) then
        begin
           d9:=StrToFloat(Edit1.Text);
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
   end;

   LoadK(Sender);

   bOk_layers:=bOk;

end;

// Загружает библиотеку материалов.
procedure TFormLayers.FormCreate(Sender: TObject);
var
  i : Integer;
begin
   bOk_layers:=false;

   ComboBox2.Clear;
   ComboBox3.Clear;
   ComboBox4.Clear;
   ComboBox5.Clear;
   ComboBox6.Clear;
   ComboBox7.Clear;
   ComboBox8.Clear;
   ComboBox9.Clear;
   ComboBox10.Clear;

   for i := 1 to Length(FormTopology.matherial) do
   begin
       ComboBox2.Items.Add(FormTopology.matherial[i-1].name);
       ComboBox3.Items.Add(FormTopology.matherial[i-1].name);
       ComboBox4.Items.Add(FormTopology.matherial[i-1].name);
       ComboBox5.Items.Add(FormTopology.matherial[i-1].name);
       ComboBox6.Items.Add(FormTopology.matherial[i-1].name);
       ComboBox7.Items.Add(FormTopology.matherial[i-1].name);
       ComboBox8.Items.Add(FormTopology.matherial[i-1].name);
       ComboBox9.Items.Add(FormTopology.matherial[i-1].name);
       ComboBox10.Items.Add(FormTopology.matherial[i-1].name);
   end;

   ComboBox2.ItemIndex:=0;
   ComboBox3.ItemIndex:=0;
   ComboBox4.ItemIndex:=0;
   ComboBox5.ItemIndex:=0;
   ComboBox6.ItemIndex:=0;
   ComboBox7.ItemIndex:=0;
   ComboBox8.ItemIndex:=0;
   ComboBox9.ItemIndex:=0;
   ComboBox10.ItemIndex:=0;
end;

end.
