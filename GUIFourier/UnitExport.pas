unit UnitExport;

interface

uses
  Winapi.Windows, Winapi.Messages, System.SysUtils, System.Variants, System.Classes, Vcl.Graphics,
  Vcl.Controls, Vcl.Forms, Vcl.Dialogs, Vcl.StdCtrls;

type
  TFormExport = class(TForm)
    CheckBox3D: TCheckBox;
    CheckBox2Dexport: TCheckBox;
    CheckBox1Dexport: TCheckBox;
    Button1: TButton;
    procedure Button1Click(Sender: TObject);
  private
    { Private declarations }
  public
    { Public declarations }
  end;

var
  FormExport: TFormExport;

implementation

{$R *.dfm}

procedure TFormExport.Button1Click(Sender: TObject);
begin
   Close;
end;

end.
