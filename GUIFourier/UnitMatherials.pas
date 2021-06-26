unit UnitMatherials;

interface

uses
  Winapi.Windows, Winapi.Messages, System.SysUtils, System.Variants, System.Classes, Vcl.Graphics,
  Vcl.Controls, Vcl.Forms, Vcl.Dialogs, Vcl.StdCtrls, Vcl.Imaging.pngimage,
  Vcl.ExtCtrls;


type
  TFormMatherials = class(TForm)
    Button1: TButton;
    Label1: TLabel;
    Edit1: TEdit;
    Label2: TLabel;
    Label3: TLabel;
    ComboBox1: TComboBox;
    GroupBox1: TGroupBox;
    Label4: TLabel;
    Label5: TLabel;
    Edit2: TEdit;
    Edit3: TEdit;
    GroupBox2: TGroupBox;
    Label6: TLabel;
    Edit4: TEdit;
    Label7: TLabel;
    Image1: TImage;
    procedure FormCreate(Sender: TObject);
    procedure ComboBox1Click(Sender: TObject);
    procedure Button1Click(Sender: TObject);
  private
    { Private declarations }
  public
    { Public declarations }


  end;

var
  FormMatherials: TFormMatherials;

implementation

{$R *.dfm}

uses Main1, UnitLayers2;

procedure TFormMatherials.Button1Click(Sender: TObject);
var
   d_check : Single;
   bOk : Boolean;
begin

     bOk:=true;
     Edit1.Color:=clWhite;
     Edit2.Color:=clWhite;
     Edit3.Color:=clWhite;
     Edit4.Color:=clWhite;

     if TryStrToFloat(Edit1.Text,d_check) then
     begin
        if (d_check>0.0) then
        begin
           FormTopology.matherial[ComboBox1.ItemIndex].thermalConductivity:=StrToFloat(Edit1.Text);
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


     if TryStrToFloat(Edit2.Text,d_check) then
     begin
        if (d_check>0.0) then
        begin
           FormTopology.matherial[ComboBox1.ItemIndex].multiplyerConductivityPlane:=StrToFloat(Edit2.Text);
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

     if TryStrToFloat(Edit3.Text,d_check) then
     begin
        if (d_check>0.0) then
        begin
           FormTopology.matherial[ComboBox1.ItemIndex].multiplyerConductivityNormal:=StrToFloat(Edit3.Text);
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


     if TryStrToFloat(Edit4.Text,d_check) then
     begin
        if (d_check<=0.0) then
        begin
           FormTopology.matherial[ComboBox1.ItemIndex].alphaForTemperatureDepend:=StrToFloat(Edit4.Text);
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


      if (bOk) then
      begin
         FormLayers.LoadK(Sender);

         Close;
      end;
end;

procedure TFormMatherials.ComboBox1Click(Sender: TObject);
begin
   // теплопроводность.
   Edit1.Text:=FloatToStr(FormTopology.matherial[ComboBox1.ItemIndex].thermalConductivity);
   // ортотропность материала.
   Edit2.Text:=FloatToStr(FormTopology.matherial[ComboBox1.ItemIndex].multiplyerConductivityPlane);
   Edit3.Text:=FloatToStr(FormTopology.matherial[ComboBox1.ItemIndex].multiplyerConductivityNormal);
   Edit4.Text:=FloatToStr(FormTopology.matherial[ComboBox1.ItemIndex].alphaForTemperatureDepend);
end;

// Загружает библиотеку материалов.
procedure TFormMatherials.FormCreate(Sender: TObject);
var
   i : Integer;
begin
   ComboBox1.Clear;
   for i := 1 to Length(FormTopology.matherial) do
   begin
       ComboBox1.Items.Add(FormTopology.matherial[i-1].name);
   end;
end;

end.
